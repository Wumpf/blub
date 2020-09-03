use crate::wgpu_utils::{self, binding_builder::*, binding_glsl, pipelines::*, shader::ShaderDirectory};
use futures::Future;
use futures::*;
use std::collections::VecDeque;
use std::rc::Rc;
use std::{path::Path, pin::Pin, time::Duration};

fn create_volume_texture_desc(label: &str, grid_dimension: wgpu::Extent3d, format: wgpu::TextureFormat) -> wgpu::TextureDescriptor {
    wgpu::TextureDescriptor {
        label: Some(label),
        size: grid_dimension,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format,
        usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::STORAGE,
    }
}

pub struct PressureSolver {
    grid_dimension: wgpu::Extent3d,

    bind_group_general: wgpu::BindGroup,
    bind_group_init: wgpu::BindGroup,
    bind_group_preconditioner: [wgpu::BindGroup; 3],
    bind_group_apply_coeff: wgpu::BindGroup,
    bind_group_dotproduct_reduce: [wgpu::BindGroup; 2],
    bind_group_dotproduct_final: [wgpu::BindGroup; 2],
    bind_group_update_pressure_and_residual: wgpu::BindGroup,
    bind_group_update_search: wgpu::BindGroup,

    pipeline_init: ComputePipelineHandle,
    pipeline_apply_preconditioner: ComputePipelineHandle,
    pipeline_reduce: ComputePipelineHandle,
    pipeline_apply_coeff: ComputePipelineHandle,
    pipeline_update_pressure_and_residual: ComputePipelineHandle,
    pipeline_update_search: ComputePipelineHandle,

    dotproduct_reduce_result_buffer: wgpu::Buffer,

    group_layout_pressure: BindGroupLayoutWithDesc,

    volume_residual_view: wgpu::TextureView,
}

const NUM_PRESSURE_ERROR_BUFFER: usize = 32;

struct PendingErrorBuffer {
    copy_operation: Option<Pin<Box<dyn Future<Output = std::result::Result<(), wgpu::BufferAsyncError>>>>>,
    buffer: wgpu::Buffer,
    resulting_sample: SolverStatisticSample,
}

pub struct SolverConfig {
    pub target_mse: f32,
    pub min_num_iterations: i32,
    pub max_num_iterations: i32,
    pub pid_config: (f32, f32, f32),
}
#[derive(Default, Copy, Clone)]
pub struct SolverStatisticSample {
    pub mse: f32,
    pub iteration_count: i32,
    timestamp: Duration,
}

// Pressure solver instance keeps track of pressure result from last step/frame in order to speed up the solve.
pub struct PressureField {
    bind_group_pressure: wgpu::BindGroup,
    volume_pressure_view: wgpu::TextureView,

    unused_error_buffers: Vec<wgpu::Buffer>,
    unscheduled_error_readbacks: Vec<PendingErrorBuffer>,
    pending_error_readbacks: VecDeque<PendingErrorBuffer>,

    pub config: SolverConfig,
    pub stats: VecDeque<SolverStatisticSample>,

    timestamp_last_iteration: Duration,
}

impl PressureField {
    const SOLVER_STATISTIC_HISTORY_LENGTH: usize = 100;
    const SOLVER_PID_INTEGRAL_HISTORY_LENGTH: usize = 8;

    pub fn new(name: &'static str, device: &wgpu::Device, grid_dimension: wgpu::Extent3d, solver: &PressureSolver, config: SolverConfig) -> Self {
        let volume_pressure = device.create_texture(&create_volume_texture_desc(
            &format!("Pressure Volume - {}", name),
            grid_dimension,
            wgpu::TextureFormat::R32Float,
        ));
        let volume_pressure_view = volume_pressure.create_view(&Default::default());

        let bind_group_pressure = BindGroupBuilder::new(&solver.group_layout_pressure)
            .texture(&volume_pressure_view)
            .create(device, &format!("BindGroup: Pressure - {}", name));

        let mut unused_error_buffers = Vec::new();
        for i in 0..NUM_PRESSURE_ERROR_BUFFER {
            unused_error_buffers.push(device.create_buffer(&wgpu::BufferDescriptor {
                size: 4,
                usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
                label: Some(&format!("Buffer: Pressure error read-back buffer {} ({})", i, name)),
                mapped_at_creation: false,
            }));
        }

        PressureField {
            bind_group_pressure,
            volume_pressure_view,
            unused_error_buffers,
            unscheduled_error_readbacks: Vec::new(),
            pending_error_readbacks: VecDeque::new(),

            config,
            stats: VecDeque::new(),

            timestamp_last_iteration: Duration::new(0, 0),
        }
    }

    pub fn pressure_view(&self) -> &wgpu::TextureView {
        &self.volume_pressure_view
    }

    fn retrieve_new_error_samples(&mut self, simulation_delta: Duration) {
        // Check if there's any new data samples
        while let Some(mut readback) = self.pending_error_readbacks.pop_front() {
            if (&mut readback.copy_operation.as_mut().unwrap()).now_or_never().is_some() {
                let mapped = readback.buffer.slice(0..4);
                let buffer_data = mapped.get_mapped_range().to_vec();
                let squared_error = *bytemuck::from_bytes::<f32>(&buffer_data);
                readback.buffer.unmap();
                self.unused_error_buffers.push(readback.buffer);

                // We currently always deal with 'pressure * density / dt', not with pressure.
                // To make display more representative for different time, we adjust our error value accordingly.
                let delta_sq = simulation_delta.as_secs_f32() * simulation_delta.as_secs_f32();
                readback.resulting_sample.mse = squared_error * delta_sq;

                self.stats.push_back(readback.resulting_sample);
                while self.stats.len() > Self::SOLVER_STATISTIC_HISTORY_LENGTH {
                    self.stats.pop_front();
                }
            } else {
                self.pending_error_readbacks.push_front(readback);
                break;
            }
        }
    }

    fn mse_to_error(&self, mse: f32) -> f32 {
        mse - self.config.target_mse
    }

    // use a PID controller to correct the solver iteration count.
    fn compute_pid_controlled_iteration_count(&self, timestep_current: Duration) -> i32 {
        let fallback_sample = SolverStatisticSample {
            iteration_count: (self.config.max_num_iterations + self.config.min_num_iterations) / 2,
            mse: self.config.target_mse,
            timestamp: Duration::new(0, 0),
        };
        let newest_sample = self.stats.back().unwrap_or(&fallback_sample);

        // integral over the last Self::SOLVER_PID_INTEGRAL_HISTORY_LENGTH deviations from target.
        let error_integral = self
            .stats
            .iter()
            .rev()
            .take(Self::SOLVER_PID_INTEGRAL_HISTORY_LENGTH)
            .map(|s| self.mse_to_error(s.mse))
            .sum::<f32>()
            / (self.stats.len().min(Self::SOLVER_PID_INTEGRAL_HISTORY_LENGTH).max(1) as f32);

        // derivative of the deviation from target.
        let previous_to_newest_sample = self.stats.iter().nth_back(1).unwrap_or(&fallback_sample);
        let error_dt = (self.mse_to_error(newest_sample.mse) - self.mse_to_error(previous_to_newest_sample.mse))
            / (newest_sample.timestamp - previous_to_newest_sample.timestamp).as_secs_f32();
        let time_since_last_sample = timestep_current - newest_sample.timestamp;

        // all components of the pid together
        let mut iteration_count = newest_sample.iteration_count;
        iteration_count += (self.config.pid_config.0
            * (self.mse_to_error(newest_sample.mse)
                + self.config.pid_config.1 * error_integral
                + self.config.pid_config.2 * time_since_last_sample.as_secs_f32() * error_dt))
            .round() as i32;

        // clamp to range
        if iteration_count < self.config.min_num_iterations {
            iteration_count = self.config.min_num_iterations;
        } else if iteration_count > self.config.max_num_iterations {
            iteration_count = self.config.max_num_iterations
        }

        iteration_count
    }

    fn enqueue_error_buffer_read(&mut self, encoder: &mut wgpu::CommandEncoder, source_buffer: &wgpu::Buffer, iteration_count: i32) {
        if let Some(target_buffer) = self.unused_error_buffers.pop() {
            encoder.copy_buffer_to_buffer(source_buffer, 0, &target_buffer, 0, 4);
            self.unscheduled_error_readbacks.push(PendingErrorBuffer {
                copy_operation: None, // Filled out in start_error_buffer_readbacks
                buffer: target_buffer,
                resulting_sample: SolverStatisticSample {
                    mse: 0.0,
                    iteration_count,
                    timestamp: self.timestamp_last_iteration,
                },
            });
        } else {
            warn!("No more error buffer available for async copy of pressure solve error");
        }
    }

    // Call this once all command
    pub fn start_error_buffer_readbacks(&mut self) {
        for mut readback in self.unscheduled_error_readbacks.drain(..) {
            readback.copy_operation = Some(readback.buffer.slice(..).map_async(wgpu::MapMode::Read).boxed());
            self.pending_error_readbacks.push_back(readback);
        }
    }
}

impl PressureSolver {
    const DOTPRODUCT_RESULTMODE_REDUCE: u32 = 0;
    const DOTPRODUCT_RESULTMODE_INIT: u32 = 1;
    const DOTPRODUCT_RESULTMODE_ALPHA: u32 = 2;
    const DOTPRODUCT_RESULTMODE_BETA: u32 = 3;

    const COMPUTE_LOCAL_SIZE_VOLUME: wgpu::Extent3d = wgpu::Extent3d {
        width: 8,
        height: 8,
        depth: 8,
    };
    const COMPUTE_LOCAL_SIZE_DOTPRODUCT_REDUCE: u32 = 1024;
    const DOTPRODUCT_READS_PER_STEP: u32 = 16; // 32 was distinctively slower, 16 about same as than 8, 4 clearly slower (gtx1070 ti)
    const DOTPRODUCT_REDUCE_REDUCTION_PER_STEP: u32 = Self::COMPUTE_LOCAL_SIZE_DOTPRODUCT_REDUCE * Self::DOTPRODUCT_READS_PER_STEP;

    pub fn new(
        device: &wgpu::Device,
        grid_dimension: wgpu::Extent3d,
        shader_dir: &ShaderDirectory,
        pipeline_manager: &mut PipelineManager,
        volume_marker_view: &wgpu::TextureView,
    ) -> Self {
        let group_layout_general = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::texture3D())
            .create(device, "BindGroupLayout: Pressure solver general");
        let group_layout_pressure = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::image3d(wgpu::TextureFormat::R32Float, false))
            .create(device, "BindGroupLayout: Pressure solver Pressure");
        let group_layout_init = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::image3d(wgpu::TextureFormat::R32Float, false))
            .create(device, "BindGroupLayout: Pressure solver init");
        let group_layout_apply_coeff = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::buffer(false))
            .next_binding_compute(binding_glsl::texture3D())
            .create(device, "BindGroupLayout: P. solver apply coeff matrix & start dot");
        let group_layout_reduce = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::buffer(true)) // source
            .next_binding_compute(binding_glsl::buffer(false)) // dest
            .create(device, "BindGroupLayout: Pressure solver dot product reduce");
        let group_layout_preconditioner = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::buffer(false))
            .next_binding_compute(binding_glsl::texture3D())
            .next_binding_compute(binding_glsl::image3d(wgpu::TextureFormat::R32Float, false))
            .next_binding_compute(binding_glsl::texture3D())
            .create(device, "BindGroupLayout: Pressure solver preconditioner");
        let group_layout_update_volume = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::buffer(false))
            .next_binding_compute(binding_glsl::image3d(wgpu::TextureFormat::R32Float, false))
            .next_binding_compute(binding_glsl::texture3D())
            .next_binding_compute(binding_glsl::uniform())
            .create(device, "BindGroupLayout: Pressure solver generic volume update");

        // Use same push constant range for all pipelines to improve internal Vulkan pipeline compatibility.
        let push_constant_ranges = &[wgpu::PushConstantRange {
            stages: wgpu::ShaderStage::COMPUTE,
            range: 0..8,
        }];

        let layout_update_volume = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Update Volume Pipeline Layout"),
            bind_group_layouts: &[
                &group_layout_general.layout,
                &group_layout_pressure.layout,
                &group_layout_update_volume.layout,
            ],
            push_constant_ranges,
        }));
        let layout_preconditioner = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pressure Solve Precondition Pipeline Layout"),
            bind_group_layouts: &[
                &group_layout_general.layout,
                &group_layout_pressure.layout,
                &group_layout_preconditioner.layout,
            ],
            push_constant_ranges,
        }));
        let layout_init = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pressure Solve Init Pipeline Layout"),
            bind_group_layouts: &[&group_layout_general.layout, &group_layout_pressure.layout, &group_layout_init.layout],
            push_constant_ranges,
        }));

        let layout_apply_coeff = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pressure Solve Apply Coeff Pipeline Layout"),
            bind_group_layouts: &[
                &group_layout_general.layout,
                &group_layout_pressure.layout,
                &group_layout_apply_coeff.layout,
            ],
            push_constant_ranges,
        }));
        let layout_reduce = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pressure Solve Reduce Pipeline Layout"),
            bind_group_layouts: &[&group_layout_general.layout, &group_layout_pressure.layout, &group_layout_reduce.layout],
            push_constant_ranges,
        }));

        let volume_residual = device.create_texture(&create_volume_texture_desc(
            "Pressure Solve Residual",
            grid_dimension,
            wgpu::TextureFormat::R32Float,
        ));
        let volume_auxiliary = device.create_texture(&create_volume_texture_desc(
            "Pressure Solve Auxiliary",
            grid_dimension,
            wgpu::TextureFormat::R32Float,
        ));
        let volume_auxiliary_temp = device.create_texture(&create_volume_texture_desc(
            "Pressure Solve Auxiliary Temp",
            grid_dimension,
            wgpu::TextureFormat::R32Float,
        ));
        let volume_search = device.create_texture(&create_volume_texture_desc(
            "Pressure Solve Search",
            grid_dimension,
            wgpu::TextureFormat::R32Float,
        ));

        let num_cells = (grid_dimension.width * grid_dimension.height * grid_dimension.depth) as u64;
        let dotproduct_reduce_step_buffers = [
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Buffer: DotProduct Reduce 0"),
                size: num_cells * std::mem::size_of::<f32>() as u64,
                usage: wgpu::BufferUsage::STORAGE,
                mapped_at_creation: false,
            }),
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Buffer: DotProduct Reduce 1"),
                size: num_cells * std::mem::size_of::<f32>() as u64 / Self::DOTPRODUCT_REDUCE_REDUCTION_PER_STEP as u64,
                usage: wgpu::BufferUsage::STORAGE,
                mapped_at_creation: false,
            }),
        ];
        let dotproduct_reduce_result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Buffer: DotProduct Result"),
            size: 4 * std::mem::size_of::<f32>() as u64,
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_SRC,
            mapped_at_creation: false,
        });

        let volume_residual_view = volume_residual.create_view(&Default::default());
        let volume_auxiliary_view = volume_auxiliary.create_view(&Default::default());
        let volume_auxiliary_temp_view = volume_auxiliary_temp.create_view(&Default::default());
        let volume_search_view = volume_search.create_view(&Default::default());

        let bind_group_general = BindGroupBuilder::new(&group_layout_general)
            .texture(&volume_marker_view)
            .create(device, "BindGroup: Pressure Solve general");
        let bind_group_init = BindGroupBuilder::new(&group_layout_init)
            .texture(&volume_residual_view)
            .create(device, "BindGroup: Compute initial residual");
        let bind_group_apply_coeff = BindGroupBuilder::new(&group_layout_apply_coeff)
            .resource(dotproduct_reduce_step_buffers[0].as_entire_binding())
            .texture(&volume_search_view)
            .create(device, "BindGroup: Apply coeff matrix & start dot product");
        let bind_group_preconditioner = [
            BindGroupBuilder::new(&group_layout_preconditioner)
                .resource(dotproduct_reduce_step_buffers[0].as_entire_binding())
                .texture(&volume_residual_view)
                .texture(&volume_auxiliary_temp_view)
                .texture(&volume_residual_view)
                .create(device, "BindGroup: Preconditioner, Step 1"),
            BindGroupBuilder::new(&group_layout_preconditioner)
                .resource(dotproduct_reduce_step_buffers[0].as_entire_binding())
                .texture(&volume_residual_view)
                .texture(&volume_auxiliary_view)
                .texture(&volume_auxiliary_temp_view)
                .create(device, "BindGroup: Preconditioner, Step 2"),
            BindGroupBuilder::new(&group_layout_preconditioner)
                .resource(dotproduct_reduce_step_buffers[0].as_entire_binding())
                .texture(&volume_residual_view)
                .texture(&volume_search_view)
                .texture(&volume_auxiliary_temp_view)
                .create(device, "BindGroup: Preconditioner, Step 2, to search"),
        ];
        let bind_group_dotproduct_reduce = [
            BindGroupBuilder::new(&group_layout_reduce)
                .resource(dotproduct_reduce_step_buffers[0].as_entire_binding())
                .resource(dotproduct_reduce_step_buffers[1].as_entire_binding())
                .create(device, "BindGroup: Pressure Solve, Reduce 0"),
            BindGroupBuilder::new(&group_layout_reduce)
                .resource(dotproduct_reduce_step_buffers[1].as_entire_binding())
                .resource(dotproduct_reduce_step_buffers[0].as_entire_binding())
                .create(device, "BindGroup: Pressure Solve, Reduce 1"),
        ];
        let bind_group_dotproduct_final = [
            BindGroupBuilder::new(&group_layout_reduce)
                .resource(dotproduct_reduce_step_buffers[0].as_entire_binding())
                .resource(dotproduct_reduce_result_buffer.as_entire_binding())
                .create(device, "BindGroup: Pressure Solve, Reduce Final 0"),
            BindGroupBuilder::new(&group_layout_reduce)
                .resource(dotproduct_reduce_step_buffers[1].as_entire_binding())
                .resource(dotproduct_reduce_result_buffer.as_entire_binding())
                .create(device, "BindGroup: Pressure Solve, Reduce Final 1"),
        ];

        let bind_group_update_pressure_and_residual = BindGroupBuilder::new(&group_layout_update_volume)
            .resource(dotproduct_reduce_step_buffers[0].as_entire_binding())
            .texture(&volume_residual_view)
            .texture(&volume_search_view)
            .resource(dotproduct_reduce_result_buffer.as_entire_binding())
            .create(device, "BindGroup: Pressure update pressure and residual");
        let bind_group_update_search = BindGroupBuilder::new(&group_layout_update_volume)
            .resource(dotproduct_reduce_step_buffers[0].as_entire_binding())
            .texture(&volume_search_view)
            .texture(&volume_auxiliary_view)
            .resource(dotproduct_reduce_result_buffer.as_entire_binding())
            .create(device, "BindGroup: Pressure update search");

        let shader_path = Path::new("simulation/pressure_solver");

        PressureSolver {
            grid_dimension: grid_dimension,

            bind_group_general,
            bind_group_init,
            bind_group_preconditioner,
            bind_group_apply_coeff,
            bind_group_dotproduct_reduce,
            bind_group_dotproduct_final,
            bind_group_update_pressure_and_residual,
            bind_group_update_search,

            pipeline_init: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(
                    "PressureSolve: Init",
                    layout_init.clone(),
                    &shader_path.join(Path::new("pressure_init.comp")),
                ),
            ),
            pipeline_apply_preconditioner: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(
                    "PressureSolve: Apply preconditioner",
                    layout_preconditioner.clone(),
                    &shader_path.join(&Path::new("pressure_apply_preconditioner.comp")),
                ),
            ),
            pipeline_reduce: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(
                    "PressureSolve: DotProduct Reduce",
                    layout_reduce.clone(),
                    &shader_path.join(&Path::new("pressure_reduce.comp")),
                ),
            ),
            pipeline_apply_coeff: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(
                    "PressureSolve: Apply coefficient matrix",
                    layout_apply_coeff.clone(),
                    &shader_path.join(&Path::new("pressure_apply_coeff.comp")),
                ),
            ),
            pipeline_update_pressure_and_residual: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(
                    "PressureSolve: Update pressure and residual",
                    layout_update_volume.clone(),
                    &shader_path.join(&Path::new("pressure_update_pressure_and_residual.comp")),
                ),
            ),
            pipeline_update_search: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(
                    "PressureSolve: Update search",
                    layout_update_volume.clone(),
                    &shader_path.join(&Path::new("pressure_update_search.comp")),
                ),
            ),

            group_layout_pressure,

            dotproduct_reduce_result_buffer,

            volume_residual_view,
        }
    }

    pub fn residual_view(&self) -> &wgpu::TextureView {
        &self.volume_residual_view
    }

    fn reduce_add<'a, 'b: 'a>(&'b self, cpass: &mut wgpu::ComputePass<'a>, pipeline_manager: &'a PipelineManager, result_mode: u32) {
        wgpu_scope!(cpass, "PressureSolver.reduce_add");

        let mut num_entries_remaining = (self.grid_dimension.width * self.grid_dimension.height * self.grid_dimension.depth) as u32;
        let mut source_buffer_index = 0;

        wgpu_scope!(cpass, "reduce", || {
            cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_reduce));
            while num_entries_remaining > Self::DOTPRODUCT_REDUCE_REDUCTION_PER_STEP {
                cpass.set_bind_group(2, &self.bind_group_dotproduct_reduce[source_buffer_index], &[]);
                cpass.set_push_constants(0, &[Self::DOTPRODUCT_RESULTMODE_REDUCE, num_entries_remaining]);
                cpass.dispatch(
                    wgpu_utils::compute_group_size_1d(
                        num_entries_remaining / Self::DOTPRODUCT_READS_PER_STEP,
                        Self::COMPUTE_LOCAL_SIZE_DOTPRODUCT_REDUCE,
                    ),
                    1,
                    1,
                );
                source_buffer_index = 1 - source_buffer_index;
                num_entries_remaining /= Self::DOTPRODUCT_REDUCE_REDUCTION_PER_STEP;
            }
        });
        wgpu_scope!(cpass, "final", || {
            cpass.set_bind_group(2, &self.bind_group_dotproduct_final[source_buffer_index], &[]);
            cpass.set_push_constants(0, &[result_mode, num_entries_remaining]);
            cpass.dispatch(
                wgpu_utils::compute_group_size_1d(num_entries_remaining, Self::COMPUTE_LOCAL_SIZE_DOTPRODUCT_REDUCE),
                1,
                1,
            );
        });
    }

    pub fn solve<'a, 'b: 'a>(
        &'b self,
        simulation_delta: Duration,
        pressure_field: &'a mut PressureField,
        encoder: &mut wgpu::CommandEncoder,
        pipeline_manager: &'a PipelineManager,
    ) {
        wgpu_scope!(encoder, "PressureSolver.solve");

        let mut cpass = encoder.begin_compute_pass();

        let grid_work_groups = wgpu_utils::compute_group_size(self.grid_dimension, Self::COMPUTE_LOCAL_SIZE_VOLUME);

        const PRECONDITIONER_PASS0: u32 = 0;
        const PRECONDITIONER_PASS1: u32 = 1;
        const PRECONDITIONER_PASS1_SET_UNUSED_TO_ZERO: u32 = 3;

        pressure_field.retrieve_new_error_samples(simulation_delta);
        let num_iterations = pressure_field.compute_pid_controlled_iteration_count(pressure_field.timestamp_last_iteration + simulation_delta);
        cpass.set_bind_group(0, &self.bind_group_general, &[]);
        cpass.set_bind_group(1, &pressure_field.bind_group_pressure, &[]);

        // For optimization various steps are collapsed as far as possible to avoid expensive buffer/texture read/writes
        // This makes the algorithm a lot faster but also a bit harder to read.

        wgpu_scope!(cpass, "init", || {
            // We use pressure from last frame, but set explicitly set all pressure values to zero wherever there is not fluid right now.
            // This is done in order to prevent having results from many frames ago influence results for upcoming frames.
            // In first step overall we instruct to use a fresh pressure buffer.
            const FIRST_STEP: u32 = 0;
            const NOT_FIRST_STEP: u32 = 1;
            cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_init));
            // Clear pressures on first step.
            // wgpu-rs doesn't zero initialize yet (bug/missing feature impl)
            // Most resources are derived from particles which we initialize ourselves, but not pressure where we use the previous step to kickstart the solver
            // https://github.com/gfx-rs/wgpu/issues/563
            if pressure_field.timestamp_last_iteration == Duration::new(0, 0) {
                cpass.set_push_constants(0, &[FIRST_STEP]);
            } else {
                cpass.set_push_constants(0, &[NOT_FIRST_STEP]);
            }
            cpass.set_bind_group(2, &self.bind_group_init, &[]);
            cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);

            // Apply preconditioner on (r), store result to search vector (s) and start dotproduct of <s; r>
            // Note that we don't use the auxillary vector here as in-between storage!
            cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_apply_preconditioner));
            cpass.set_push_constants(0, &[0]);
            cpass.set_push_constants(0, &[PRECONDITIONER_PASS0]);
            cpass.set_bind_group(2, &self.bind_group_preconditioner[0], &[]);
            cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);
            cpass.set_push_constants(0, &[PRECONDITIONER_PASS1_SET_UNUSED_TO_ZERO]);
            cpass.set_bind_group(2, &self.bind_group_preconditioner[2], &[]);
            cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);
            // Init sigma to dotproduct of search vector (s) and residual (r)
            self.reduce_add(&mut cpass, pipeline_manager, Self::DOTPRODUCT_RESULTMODE_INIT);
        });

        wgpu_scope!(cpass, "solver iterations", || {
            let mut i = 0;
            while wgpu_scope!(cpass, &format!("iteration {}", i), || {
                // Apply cell relationships to search vector (i.e. multiply s with A)
                // The dot product is applied to the result (denoted as z in Bridson's book) and the search vector (s), i.e. compute <s; As>
                cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_apply_coeff));
                cpass.set_bind_group(2, &self.bind_group_apply_coeff, &[]);
                cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);
                // finish dotproduct of auxiliary field (z) and search field (s)
                self.reduce_add(&mut cpass, pipeline_manager, Self::DOTPRODUCT_RESULTMODE_ALPHA);

                wgpu_scope!(cpass, "update pressure field (p) and residual field (r)", || {
                    const PRUPDATE_LAST_ITERATION: u32 = 1;
                    cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_update_pressure_and_residual));
                    if i == num_iterations {
                        cpass.set_push_constants(0, &[PRUPDATE_LAST_ITERATION]);
                    } else {
                        cpass.set_push_constants(0, &[0]);
                    }
                    cpass.set_bind_group(2, &self.bind_group_update_pressure_and_residual, &[]);
                    cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);
                });

                // Was this the last update?
                if i == num_iterations {
                    // Compute remaining error, we use this later to feedback to the number of iterations.
                    self.reduce_add(&mut cpass, pipeline_manager, Self::DOTPRODUCT_RESULTMODE_REDUCE);
                    return false;
                }

                // Apply preconditioner on (r), store result to auxillary (z) and start dotproduct of <z; r>
                cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_apply_preconditioner));
                cpass.set_push_constants(0, &[PRECONDITIONER_PASS0]);
                cpass.set_bind_group(2, &self.bind_group_preconditioner[0], &[]);
                cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);
                cpass.set_push_constants(0, &[PRECONDITIONER_PASS1]);
                cpass.set_bind_group(2, &self.bind_group_preconditioner[1], &[]);
                cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);
                // finish dotproduct of auxiliary field (z) and residual field (r)
                self.reduce_add(&mut cpass, pipeline_manager, Self::DOTPRODUCT_RESULTMODE_BETA);

                // Update search vector
                cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_update_search));
                cpass.set_bind_group(2, &self.bind_group_update_search, &[]);
                cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);

                i += 1;
                true
            }) {}
        });

        drop(cpass);
        pressure_field.timestamp_last_iteration += simulation_delta;
        pressure_field.enqueue_error_buffer_read(&mut *encoder, &self.dotproduct_reduce_result_buffer, num_iterations);
    }
}
