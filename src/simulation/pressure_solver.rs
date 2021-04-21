use crate::wgpu_utils::{self, binding_builder::*, binding_glsl, pipelines::*, shader::ShaderDirectory};
use futures::Future;
use futures::*;
use std::collections::VecDeque;
use std::rc::Rc;
use std::{path::Path, pin::Pin, time::Duration};
use wgpu_profiler::{wgpu_profiler, GpuProfiler};
use wgpu_utils::uniformbuffer::UniformBuffer;

fn create_volume_texture_desc(label: &str, grid_dimension: wgpu::Extent3d, format: wgpu::TextureFormat) -> wgpu::TextureDescriptor {
    wgpu::TextureDescriptor {
        label: Some(label),
        size: grid_dimension,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format,
        usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::STORAGE | wgpu::TextureUsage::COPY_DST,
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
    pipeline_reduce_sum: ComputePipelineHandle,
    pipeline_reduce_max: ComputePipelineHandle,
    pipeline_apply_coeff: ComputePipelineHandle,
    pipeline_update_pressure_and_residual: ComputePipelineHandle,
    pipeline_update_search: ComputePipelineHandle,

    dotproduct_reduce_result_and_dispatch_buffer: wgpu::Buffer,

    group_layout_pressure_field: BindGroupLayoutWithDesc,

    volume_residual_view: wgpu::TextureView,
}

const NUM_PRESSURE_ERROR_BUFFER: usize = 32;

struct PendingErrorBuffer {
    copy_operation: Option<Pin<Box<dyn Future<Output = std::result::Result<(), wgpu::BufferAsyncError>>>>>,
    buffer: wgpu::Buffer,
    resulting_sample: SolverStatisticSample,
}

#[derive(Copy, Clone)]
pub struct SolverConfig {
    pub error_tolerance: f32,
    pub max_num_iterations: i32,
    pub error_check_frequency: i32,
}
#[derive(Default, Copy, Clone)]
pub struct SolverStatisticSample {
    pub error: f32,
    pub iteration_count: i32,
    //timestamp: Duration,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SolverConfigUniformBufferContent {
    // We compute internally the max error for a pseudo pressure value defined as 'pressure * density / dt', not with pressure.
    // For easier handling with different timesteps the user facing parameter is about 'pressure * density'.
    error_tolerance: f32,
    max_num_iterations: u32,
}
unsafe impl bytemuck::Pod for SolverConfigUniformBufferContent {}
unsafe impl bytemuck::Zeroable for SolverConfigUniformBufferContent {}

type SolverConfigUniformBuffer = UniformBuffer<SolverConfigUniformBufferContent>;

// Pressure solver instance keeps track of pressure result from last step/frame in order to speed up the solve.
pub struct PressureField {
    bind_group_pressure_field: wgpu::BindGroup,
    volume_pressure: wgpu::Texture,
    volume_pressure_view: wgpu::TextureView,

    unused_error_buffers: Vec<wgpu::Buffer>,
    unscheduled_error_readbacks: Vec<PendingErrorBuffer>,
    pending_error_readbacks: VecDeque<PendingErrorBuffer>,

    config_ubo: SolverConfigUniformBuffer,
    pub config: SolverConfig,
    pub stats: VecDeque<SolverStatisticSample>,

    timestamp_last_iteration: Duration,
}

impl PressureField {
    const SOLVER_STATISTIC_HISTORY_LENGTH: usize = 100;

    pub fn new(name: &'static str, device: &wgpu::Device, grid_dimension: wgpu::Extent3d, solver: &PressureSolver, config: SolverConfig) -> Self {
        let volume_pressure = device.create_texture(&create_volume_texture_desc(
            &format!("Pressure Volume - {}", name),
            grid_dimension,
            wgpu::TextureFormat::R32Float,
        ));
        let volume_pressure_view = volume_pressure.create_view(&Default::default());

        let config_ubo = SolverConfigUniformBuffer::new(device);

        let bind_group_pressure_field = BindGroupBuilder::new(&solver.group_layout_pressure_field)
            .texture(&volume_pressure_view)
            .resource(config_ubo.binding_resource())
            .create(device, &format!("BindGroup: Pressure - {}", name));

        let mut unused_error_buffers = Vec::new();
        for i in 0..NUM_PRESSURE_ERROR_BUFFER {
            unused_error_buffers.push(device.create_buffer(&wgpu::BufferDescriptor {
                size: 8,
                usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
                label: Some(&format!("Buffer: Pressure error read-back buffer {} ({})", i, name)),
                mapped_at_creation: false,
            }));
        }

        PressureField {
            bind_group_pressure_field,
            volume_pressure,
            volume_pressure_view,
            unused_error_buffers,
            unscheduled_error_readbacks: Vec::new(),
            pending_error_readbacks: VecDeque::new(),

            config_ubo,
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
                let mapped = readback.buffer.slice(0..8);
                let buffer_data = mapped.get_mapped_range().to_vec();
                let max_error = *bytemuck::from_bytes::<f32>(&buffer_data[0..4]);
                let iteration_count = *bytemuck::from_bytes::<f32>(&buffer_data[4..8]);
                readback.buffer.unmap();
                self.unused_error_buffers.push(readback.buffer);

                // We always deal with 'pressure * dt / density' in the solver, not with pressure.
                // To make display more representative for different time, we adjust our error value accordingly.
                // See also config.error_tolerance
                readback.resulting_sample.error = max_error * simulation_delta.as_secs_f32();
                readback.resulting_sample.iteration_count = iteration_count as i32;

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

    fn enqueue_error_buffer_read(&mut self, encoder: &mut wgpu::CommandEncoder, source_buffer: &wgpu::Buffer) {
        if let Some(target_buffer) = self.unused_error_buffers.pop() {
            encoder.copy_buffer_to_buffer(source_buffer, 8, &target_buffer, 0, 8);
            self.unscheduled_error_readbacks.push(PendingErrorBuffer {
                copy_operation: None, // Filled out in start_error_buffer_readbacks
                buffer: target_buffer,
                resulting_sample: SolverStatisticSample {
                    error: 0.0,
                    iteration_count: 0,
                    //timestamp: self.timestamp_last_iteration,
                },
            });
        } else {
            warn!("No more error buffer available for async copy of pressure solve error");
        }
    }

    pub fn update_uniforms(&mut self, queue: &wgpu::Queue, simulation_delta: Duration) {
        self.config_ubo.update_content(
            queue,
            SolverConfigUniformBufferContent {
                error_tolerance: self.config.error_tolerance / simulation_delta.as_secs_f32(),
                max_num_iterations: self.config.max_num_iterations as u32,
            },
        );
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
    const REDUCE_RESULTMODE_REDUCE: u32 = 0;
    const REDUCE_RESULTMODE_INIT: u32 = 1;
    const REDUCE_RESULTMODE_ALPHA: u32 = 2;
    const REDUCE_RESULTMODE_BETA: u32 = 3;
    const REDUCE_RESULTMODE_MAX_ERROR: u32 = 4;

    const COMPUTE_LOCAL_SIZE_VOLUME: wgpu::Extent3d = wgpu::Extent3d {
        width: 8,
        height: 8,
        depth_or_array_layers: 1,
    };
    const COMPUTE_LOCAL_SIZE_REDUCE: u32 = 1024;
    const REDUCE_READS_PER_THREAD: u32 = 16; // 32 was distinctively slower, 16 about same as than 8, 4 clearly slower (gtx1070 ti)
    const REDUCE_REDUCTION_PER_STEP: u32 = Self::COMPUTE_LOCAL_SIZE_REDUCE * Self::REDUCE_READS_PER_THREAD;

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
        let group_layout_pressure_field = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::image3D(
                wgpu::TextureFormat::R32Float,
                wgpu::StorageTextureAccess::ReadWrite,
            ))
            .next_binding_compute(binding_glsl::uniform())
            .create(device, "BindGroupLayout: Pressure solver Pressure");
        let group_layout_init = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::image3D(
                wgpu::TextureFormat::R32Float,
                wgpu::StorageTextureAccess::ReadWrite,
            ))
            .next_binding_compute(binding_glsl::buffer(false))
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
            .next_binding_compute(binding_glsl::image3D(
                wgpu::TextureFormat::R32Float,
                wgpu::StorageTextureAccess::ReadWrite,
            ))
            .next_binding_compute(binding_glsl::texture3D())
            .create(device, "BindGroupLayout: Pressure solver preconditioner");
        let group_layout_update_volume = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::buffer(false))
            .next_binding_compute(binding_glsl::image3D(
                wgpu::TextureFormat::R32Float,
                wgpu::StorageTextureAccess::ReadWrite,
            ))
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
                &group_layout_pressure_field.layout,
                &group_layout_update_volume.layout,
            ],
            push_constant_ranges,
        }));
        let layout_preconditioner = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pressure Solve Precondition Pipeline Layout"),
            bind_group_layouts: &[
                &group_layout_general.layout,
                &group_layout_pressure_field.layout,
                &group_layout_preconditioner.layout,
            ],
            push_constant_ranges,
        }));
        let layout_init = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pressure Solve Init Pipeline Layout"),
            bind_group_layouts: &[
                &group_layout_general.layout,
                &group_layout_pressure_field.layout,
                &group_layout_init.layout,
            ],
            push_constant_ranges,
        }));

        let layout_apply_coeff = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pressure Solve Apply Coeff Pipeline Layout"),
            bind_group_layouts: &[
                &group_layout_general.layout,
                &group_layout_pressure_field.layout,
                &group_layout_apply_coeff.layout,
            ],
            push_constant_ranges,
        }));
        let layout_reduce = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pressure Solve Reduce Pipeline Layout"),
            bind_group_layouts: &[
                &group_layout_general.layout,
                &group_layout_pressure_field.layout,
                &group_layout_reduce.layout,
            ],
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

        let num_cells = (grid_dimension.width * grid_dimension.height * grid_dimension.depth_or_array_layers) as u64;
        let dotproduct_reduce_step_buffers = [
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Buffer: DotProduct Reduce 0"),
                size: num_cells * std::mem::size_of::<f32>() as u64,
                usage: wgpu::BufferUsage::STORAGE,
                mapped_at_creation: false,
            }),
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Buffer: DotProduct Reduce 1"),
                size: num_cells * std::mem::size_of::<f32>() as u64 / Self::REDUCE_REDUCTION_PER_STEP as u64,
                usage: wgpu::BufferUsage::STORAGE,
                mapped_at_creation: false,
            }),
        ];
        let dotproduct_reduce_result_and_dispatch_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Buffer: DotProduct Result & IndirectDispatch buffer"),
            size: 16 * std::mem::size_of::<f32>() as u64,
            usage: wgpu::BufferUsage::INDIRECT | wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_SRC,
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
            .resource(dotproduct_reduce_result_and_dispatch_buffer.as_entire_binding())
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
                .resource(dotproduct_reduce_result_and_dispatch_buffer.as_entire_binding())
                .create(device, "BindGroup: Pressure Solve, Reduce Final 0"),
            BindGroupBuilder::new(&group_layout_reduce)
                .resource(dotproduct_reduce_step_buffers[1].as_entire_binding())
                .resource(dotproduct_reduce_result_and_dispatch_buffer.as_entire_binding())
                .create(device, "BindGroup: Pressure Solve, Reduce Final 1"),
        ];

        let bind_group_update_pressure_and_residual = BindGroupBuilder::new(&group_layout_update_volume)
            .resource(dotproduct_reduce_step_buffers[0].as_entire_binding())
            .texture(&volume_residual_view)
            .texture(&volume_search_view)
            .resource(dotproduct_reduce_result_and_dispatch_buffer.as_entire_binding())
            .create(device, "BindGroup: Pressure update pressure and residual");
        let bind_group_update_search = BindGroupBuilder::new(&group_layout_update_volume)
            .resource(dotproduct_reduce_step_buffers[0].as_entire_binding())
            .texture(&volume_search_view)
            .texture(&volume_auxiliary_view)
            .resource(dotproduct_reduce_result_and_dispatch_buffer.as_entire_binding())
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
            pipeline_reduce_sum: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(
                    "PressureSolve: DotProduct Reduce",
                    layout_reduce.clone(),
                    &shader_path.join(&Path::new("pressure_reduce_sum.comp")),
                ),
            ),
            pipeline_reduce_max: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(
                    "PressureSolve: DotProduct Reduce",
                    layout_reduce.clone(),
                    &shader_path.join(&Path::new("pressure_reduce_max.comp")),
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

            group_layout_pressure_field,

            dotproduct_reduce_result_and_dispatch_buffer,

            volume_residual_view,
        }
    }

    pub fn residual_view(&self) -> &wgpu::TextureView {
        &self.volume_residual_view
    }

    fn reduce_add<'a, 'b: 'a>(&'b self, cpass: &mut wgpu::ComputePass<'a>, pipeline_manager: &'a PipelineManager, result_mode: u32) {
        self.reduce(cpass, pipeline_manager, result_mode, &self.pipeline_reduce_sum);
    }

    fn reduce_max<'a, 'b: 'a>(&'b self, cpass: &mut wgpu::ComputePass<'a>, pipeline_manager: &'a PipelineManager, result_mode: u32) {
        self.reduce(cpass, pipeline_manager, result_mode, &self.pipeline_reduce_max);
    }

    fn reduce<'a, 'b: 'a>(
        &'b self,
        cpass: &mut wgpu::ComputePass<'a>,
        pipeline_manager: &'a PipelineManager,
        result_mode: u32,
        pipeline: &ComputePipelineHandle,
    ) {
        let mut num_entries_remaining = (self.grid_dimension.width * self.grid_dimension.height * self.grid_dimension.depth_or_array_layers) as u32;
        assert!(num_entries_remaining > Self::REDUCE_REDUCTION_PER_STEP);
        let mut source_buffer_index = 0;

        // the first few reduce steps are indirect dispatches so we can disable them if we reached some error threshold.
        const DISPATCH_BUFFER_OFFSETS: [u64; 2] = [(4 * 4) * 2, (4 * 4) * 2];

        // Reduce
        // Even if we have a 512x512x512 volume, this step will only run twice with the config we have right now (16384 elements reduced per block)
        cpass.set_pipeline(pipeline_manager.get_compute(pipeline));
        let mut reduce_step_idx = 0;
        while num_entries_remaining > Self::REDUCE_REDUCTION_PER_STEP {
            cpass.set_bind_group(2, &self.bind_group_dotproduct_reduce[source_buffer_index], &[]);
            cpass.set_push_constants(0, &bytemuck::bytes_of(&[Self::REDUCE_RESULTMODE_REDUCE, num_entries_remaining]));

            if reduce_step_idx < DISPATCH_BUFFER_OFFSETS.len() {
                cpass.dispatch_indirect(
                    &self.dotproduct_reduce_result_and_dispatch_buffer,
                    DISPATCH_BUFFER_OFFSETS[reduce_step_idx],
                )
            } else {
                cpass.dispatch(
                    wgpu_utils::compute_group_size_1d(num_entries_remaining / Self::REDUCE_READS_PER_THREAD, Self::COMPUTE_LOCAL_SIZE_REDUCE),
                    1,
                    1,
                );
            }
            source_buffer_index = 1 - source_buffer_index;
            num_entries_remaining /= Self::REDUCE_REDUCTION_PER_STEP;

            reduce_step_idx += 1;
        }

        // Final.
        // Right now not a dispatch_indirect, so we always run it even if we decided that it is no longer necessary.
        // It's simply a bit too tricky to turn it off - we can't write into a dispatch buffer that is in use
        cpass.set_bind_group(2, &self.bind_group_dotproduct_final[source_buffer_index], &[]);
        cpass.set_push_constants(0, &bytemuck::bytes_of(&[result_mode, num_entries_remaining]));
        cpass.dispatch(1, 1, 1);
    }

    pub fn solve<'a, 'b: 'a>(
        &'b self,
        simulation_delta: Duration,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        pressure_field: &'a mut PressureField,
        pipeline_manager: &'a PipelineManager,
        profiler: &mut GpuProfiler,
    ) {
        // Clear pressures on first overall step of this pressure field.
        if pressure_field.timestamp_last_iteration == Duration::new(0, 0) {
            encoder.clear_texture(&pressure_field.volume_pressure, &Default::default());
        }

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pressure solve"),
        });

        const PRECONDITIONER_PASS0: u32 = 0;
        const PRECONDITIONER_PASS1: u32 = 1;

        pressure_field.retrieve_new_error_samples(simulation_delta);

        let reduce_pass_initial_group_size = wgpu_utils::compute_group_size_1d(
            (self.grid_dimension.width * self.grid_dimension.height * self.grid_dimension.depth_or_array_layers) as u32
                / Self::REDUCE_READS_PER_THREAD,
            Self::COMPUTE_LOCAL_SIZE_REDUCE,
        );

        cpass.set_bind_group(0, &self.bind_group_general, &[]);
        cpass.set_bind_group(1, &pressure_field.bind_group_pressure_field, &[]);

        // For optimization various steps are collapsed as far as possible to avoid expensive buffer/texture read/writes
        // This makes the algorithm a lot faster but also a bit harder to read.
        wgpu_profiler!("init", profiler, &mut cpass, device, {
            let grid_work_groups = wgpu_utils::compute_group_size(self.grid_dimension, Self::COMPUTE_LOCAL_SIZE_VOLUME);

            // We use pressure from last frame, but set explicitly set all pressure values to zero wherever there is not fluid right now.
            // This is done in order to prevent having results from many frames ago influence results for upcoming frames.
            cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_init));
            cpass.set_bind_group(2, &self.bind_group_init, &[]);
            cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth_or_array_layers);

            // Apply preconditioner on (r), store result to search vector (s) and start dotproduct of <s; r>
            // Note that we don't use the auxillary vector here as in-between storage!
            wgpu_profiler!("preconditioner(r) ➡ s, start s·r", profiler, &mut cpass, device, {
                cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_apply_preconditioner));
                cpass.set_push_constants(0, &bytemuck::bytes_of(&[0 as u32]));
                cpass.set_push_constants(0, &bytemuck::bytes_of(&[PRECONDITIONER_PASS0]));
                cpass.set_bind_group(2, &self.bind_group_preconditioner[0], &[]);
                cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth_or_array_layers);
                cpass.set_push_constants(0, &bytemuck::bytes_of(&[PRECONDITIONER_PASS1, reduce_pass_initial_group_size]));
                cpass.set_bind_group(2, &self.bind_group_preconditioner[2], &[]);
                cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth_or_array_layers);
            });
            wgpu_profiler!("reduce_add: finish s·r ➡ sigma", profiler, &mut cpass, device, {
                self.reduce_add(&mut cpass, pipeline_manager, Self::REDUCE_RESULTMODE_INIT);
            });
        });

        wgpu_profiler!("solver iterations", profiler, &mut cpass, device, {
            const DISPATCH_BUFFER_OFFSET: u64 = 4 * 4;

            let mut i = 0;
            while wgpu_profiler!(
                &format!("iteration {}", i),
                profiler,
                &mut cpass,
                device,
                (|| {
                    wgpu_profiler!("sA ➡ z, start s·z", profiler, &mut cpass, device, {
                        // The dot product is applied to the result (denoted as z in Bridson's book) and the search vector (s), i.e. compute <s; As>
                        cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_apply_coeff));
                        cpass.set_bind_group(2, &self.bind_group_apply_coeff, &[]);
                        cpass.set_push_constants(0, &bytemuck::bytes_of(&[0, reduce_pass_initial_group_size]));
                        cpass.dispatch_indirect(&self.dotproduct_reduce_result_and_dispatch_buffer, DISPATCH_BUFFER_OFFSET);
                    });
                    wgpu_profiler!("reduce_add: finish s·z ➡ alpha", profiler, &mut cpass, device, {
                        self.reduce_add(&mut cpass, pipeline_manager, Self::REDUCE_RESULTMODE_ALPHA);
                    });

                    let iteration_with_error_computation =
                        pressure_field.config.max_num_iterations == i || (i > 0 && i % pressure_field.config.error_check_frequency == 0);

                    wgpu_profiler!("update pressure field (p) & residual field (r)", profiler, &mut cpass, device, {
                        const PRUPDATE_COMPUTE_MAX_ERROR: u32 = 1;
                        cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_update_pressure_and_residual));
                        if iteration_with_error_computation {
                            cpass.set_push_constants(0, &bytemuck::bytes_of(&[PRUPDATE_COMPUTE_MAX_ERROR, reduce_pass_initial_group_size]));
                        } else {
                            cpass.set_push_constants(0, &bytemuck::bytes_of(&[0]));
                        }
                        cpass.set_bind_group(2, &self.bind_group_update_pressure_and_residual, &[]);
                        cpass.dispatch_indirect(&self.dotproduct_reduce_result_and_dispatch_buffer, DISPATCH_BUFFER_OFFSET);
                    });

                    // Time to check on error?
                    if iteration_with_error_computation {
                        // Compute remaining error.
                        // Used for statistics. If below target, makes all upcoming dispatch_indirect no-ops.
                        wgpu_profiler!("reduce: compute max error", profiler, &mut cpass, device, {
                            self.reduce_max(&mut cpass, pipeline_manager, Self::REDUCE_RESULTMODE_MAX_ERROR + i as u32);
                        });

                        if pressure_field.config.max_num_iterations == i {
                            return false;
                        }
                    }

                    wgpu_profiler!("preconditioner(r) ➡ (z), start z·r", profiler, &mut cpass, device, {
                        cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_apply_preconditioner));
                        cpass.set_push_constants(0, &bytemuck::bytes_of(&[PRECONDITIONER_PASS0]));
                        cpass.set_bind_group(2, &self.bind_group_preconditioner[0], &[]);
                        cpass.dispatch_indirect(&self.dotproduct_reduce_result_and_dispatch_buffer, DISPATCH_BUFFER_OFFSET);
                        cpass.set_push_constants(0, &bytemuck::bytes_of(&[PRECONDITIONER_PASS1, reduce_pass_initial_group_size]));
                        cpass.set_bind_group(2, &self.bind_group_preconditioner[1], &[]);
                        cpass.dispatch_indirect(&self.dotproduct_reduce_result_and_dispatch_buffer, DISPATCH_BUFFER_OFFSET);
                    });

                    wgpu_profiler!("reduce_add: finish z·r ➡ beta", profiler, &mut cpass, device, {
                        self.reduce_add(&mut cpass, pipeline_manager, Self::REDUCE_RESULTMODE_BETA);
                    });

                    wgpu_profiler!("Update search vector (s)", profiler, &mut cpass, device, {
                        cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_update_search));
                        cpass.set_bind_group(2, &self.bind_group_update_search, &[]);
                        cpass.dispatch_indirect(&self.dotproduct_reduce_result_and_dispatch_buffer, DISPATCH_BUFFER_OFFSET);
                    });

                    i += 1;
                    true
                })()
            ) {}
        });

        drop(cpass);
        pressure_field.timestamp_last_iteration += simulation_delta;
        pressure_field.enqueue_error_buffer_read(&mut *encoder, &self.dotproduct_reduce_result_and_dispatch_buffer);
    }
}
