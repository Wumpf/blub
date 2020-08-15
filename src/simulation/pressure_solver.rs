use crate::wgpu_utils::{self, binding_builder::*, binding_glsl, pipelines::*, shader::ShaderDirectory};
use std::{cell::Cell, path::Path, rc::Rc};

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

    volume_pressure_view: wgpu::TextureView,
    volume_pcg_residual_view: wgpu::TextureView,

    is_first_step: Cell<bool>,
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
        let group_layout_pressure_solve_init = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::image3d(wgpu::TextureFormat::R32Float, false))
            .next_binding_compute(binding_glsl::image3d(wgpu::TextureFormat::R32Float, false))
            .create(device, "BindGroupLayout: Pressure solver init");
        let group_layout_apply_coeff = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::buffer(false))
            .next_binding_compute(binding_glsl::texture3D())
            .create(device, "BindGroupLayout: P. solver apply coeff matrix & start dot");
        let group_layout_pressure_solve_reduce = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::buffer(true)) // source
            .next_binding_compute(binding_glsl::buffer(false)) // dest
            .create(device, "BindGroupLayout: Pressure solver dot product reduce");
        let group_layout_pressure_preconditioner = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::buffer(false))
            .next_binding_compute(binding_glsl::texture3D())
            .next_binding_compute(binding_glsl::image3d(wgpu::TextureFormat::R32Float, false))
            .next_binding_compute(binding_glsl::texture3D())
            .create(device, "BindGroupLayout: Pressure solver preconditioner");
        let group_layout_pressure_update_volume = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::image3d(wgpu::TextureFormat::R32Float, false))
            .next_binding_compute(binding_glsl::texture3D())
            .next_binding_compute(binding_glsl::uniform())
            .create(device, "BindGroupLayout: Pressure solver generic volume update");
        let group_layout_pressure_update_pressure_and_residual = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::image3d(wgpu::TextureFormat::R32Float, false))
            .next_binding_compute(binding_glsl::image3d(wgpu::TextureFormat::R32Float, false))
            .next_binding_compute(binding_glsl::texture3D())
            .next_binding_compute(binding_glsl::uniform())
            .create(device, "BindGroupLayout: Pressure solver update pressure and residual");

        // Use same push constant range for all pipelines to improve internal Vulkan pipeline compatibility.
        let push_constant_ranges = &[wgpu::PushConstantRange {
            stages: wgpu::ShaderStage::COMPUTE,
            range: 0..8,
        }];

        let layout_pressure_update_volume = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Update Volume Pipeline Layout"),
            bind_group_layouts: &[&group_layout_general.layout, &group_layout_pressure_update_volume.layout],
            push_constant_ranges,
        }));
        let layout_pressure_preconditioner = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pressure Solve Precondition Pipeline Layout"),
            bind_group_layouts: &[&group_layout_general.layout, &group_layout_pressure_preconditioner.layout],
            push_constant_ranges,
        }));
        let layout_pressure_update_pressure_and_residual = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pressure Solve Update P&R Pipeline Layout"),
            bind_group_layouts: &[&group_layout_general.layout, &group_layout_pressure_update_pressure_and_residual.layout],
            push_constant_ranges,
        }));
        let layout_pressure_solve_init = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pressure Solve Init Pipeline Layout"),
            bind_group_layouts: &[&group_layout_general.layout, &group_layout_pressure_solve_init.layout],
            push_constant_ranges,
        }));

        let layout_pressure_apply_coeff = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pressure Solve Apply Coeff Pipeline Layout"),
            bind_group_layouts: &[&group_layout_general.layout, &group_layout_apply_coeff.layout],
            push_constant_ranges,
        }));
        let layout_pressure_solve_reduce = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pressure Solve Reduce Pipeline Layout"),
            bind_group_layouts: &[&group_layout_general.layout, &group_layout_pressure_solve_reduce.layout],
            push_constant_ranges,
        }));

        let create_volume_texture_desc = |label: &'static str, format: wgpu::TextureFormat| -> wgpu::TextureDescriptor {
            wgpu::TextureDescriptor {
                label: Some(label),
                size: grid_dimension,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format,
                usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::STORAGE,
            }
        };

        let volume_pressure_from_velocity = device.create_texture(&create_volume_texture_desc("Pressure Volume", wgpu::TextureFormat::R32Float)); // Previous frame is required for this one!
        let volume_pcg_residual = device.create_texture(&create_volume_texture_desc("Pressure Solve Residual", wgpu::TextureFormat::R32Float));
        let volume_pcg_auxiliary = device.create_texture(&create_volume_texture_desc("Pressure Solve Auxiliary", wgpu::TextureFormat::R32Float));
        let volume_pcg_auxiliary_temp = device.create_texture(&create_volume_texture_desc(
            "Pressure Solve Auxiliary Temp",
            wgpu::TextureFormat::R32Float,
        ));
        let volume_pcg_search = device.create_texture(&create_volume_texture_desc("Pressure Solve Search", wgpu::TextureFormat::R32Float));

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
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::UNIFORM,
            mapped_at_creation: false,
        });

        let volume_pressure_from_velocity_view = volume_pressure_from_velocity.create_view(&Default::default());
        let volume_pcg_residual_view = volume_pcg_residual.create_view(&Default::default());
        let volume_pcg_auxiliary_view = volume_pcg_auxiliary.create_view(&Default::default());
        let volume_pcg_auxiliary_temp_view = volume_pcg_auxiliary_temp.create_view(&Default::default());
        let volume_pcg_search_view = volume_pcg_search.create_view(&Default::default());

        let bind_group_general = BindGroupBuilder::new(&group_layout_general)
            .texture(&volume_marker_view)
            .create(device, "BindGroup: Pressure Solve general");
        let bind_group_init = BindGroupBuilder::new(&group_layout_pressure_solve_init)
            .texture(&volume_pcg_residual_view)
            .texture(&volume_pressure_from_velocity_view)
            .create(device, "BindGroup: Compute initial residual");
        let bind_group_apply_coeff = BindGroupBuilder::new(&group_layout_apply_coeff)
            .buffer(dotproduct_reduce_step_buffers[0].slice(..))
            .texture(&volume_pcg_search_view)
            .create(device, "BindGroup: Apply coeff matrix & start dot product");
        let bind_group_preconditioner = [
            BindGroupBuilder::new(&group_layout_pressure_preconditioner)
                .buffer(dotproduct_reduce_step_buffers[0].slice(..))
                .texture(&volume_pcg_residual_view)
                .texture(&volume_pcg_auxiliary_temp_view)
                .texture(&volume_pcg_residual_view)
                .create(device, "BindGroup: Preconditioner, Step 1"),
            BindGroupBuilder::new(&group_layout_pressure_preconditioner)
                .buffer(dotproduct_reduce_step_buffers[0].slice(..))
                .texture(&volume_pcg_residual_view)
                .texture(&volume_pcg_auxiliary_view)
                .texture(&volume_pcg_auxiliary_temp_view)
                .create(device, "BindGroup: Preconditioner, Step 2"),
            BindGroupBuilder::new(&group_layout_pressure_preconditioner)
                .buffer(dotproduct_reduce_step_buffers[0].slice(..))
                .texture(&volume_pcg_residual_view)
                .texture(&volume_pcg_search_view)
                .texture(&volume_pcg_auxiliary_temp_view)
                .create(device, "BindGroup: Preconditioner, Step 2, to search"),
        ];
        let bind_group_dotproduct_reduce = [
            BindGroupBuilder::new(&group_layout_pressure_solve_reduce)
                .buffer(dotproduct_reduce_step_buffers[0].slice(..))
                .buffer(dotproduct_reduce_step_buffers[1].slice(..))
                .create(device, "BindGroup: Pressure Solve, Reduce 0"),
            BindGroupBuilder::new(&group_layout_pressure_solve_reduce)
                .buffer(dotproduct_reduce_step_buffers[1].slice(..))
                .buffer(dotproduct_reduce_step_buffers[0].slice(..))
                .create(device, "BindGroup: Pressure Solve, Reduce 1"),
        ];
        let bind_group_dotproduct_final = [
            BindGroupBuilder::new(&group_layout_pressure_solve_reduce)
                .buffer(dotproduct_reduce_step_buffers[0].slice(..))
                .buffer(dotproduct_reduce_result_buffer.slice(..))
                .create(device, "BindGroup: Pressure Solve, Reduce Final 0"),
            BindGroupBuilder::new(&group_layout_pressure_solve_reduce)
                .buffer(dotproduct_reduce_step_buffers[1].slice(..))
                .buffer(dotproduct_reduce_result_buffer.slice(..))
                .create(device, "BindGroup: Pressure Solve, Reduce Final 1"),
        ];

        let bind_group_update_pressure_and_residual = BindGroupBuilder::new(&group_layout_pressure_update_pressure_and_residual)
            .texture(&volume_pressure_from_velocity_view)
            .texture(&volume_pcg_residual_view)
            .texture(&volume_pcg_search_view)
            .buffer(dotproduct_reduce_result_buffer.slice(..))
            .create(device, "BindGroup: Pressure update pressure and residual");
        let bind_group_update_search = BindGroupBuilder::new(&group_layout_pressure_update_volume)
            .texture(&volume_pcg_search_view)
            .texture(&volume_pcg_auxiliary_view)
            .buffer(dotproduct_reduce_result_buffer.slice(..))
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
                ComputePipelineCreationDesc::new(layout_pressure_solve_init.clone(), &shader_path.join(Path::new("pressure_init.comp"))),
            ),
            pipeline_apply_preconditioner: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(
                    layout_pressure_preconditioner.clone(),
                    &shader_path.join(&Path::new("pressure_apply_preconditioner.comp")),
                ),
            ),
            pipeline_reduce: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(
                    layout_pressure_solve_reduce.clone(),
                    &shader_path.join(&Path::new("pressure_reduce.comp")),
                ),
            ),
            pipeline_apply_coeff: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(
                    layout_pressure_apply_coeff.clone(),
                    &shader_path.join(&Path::new("pressure_apply_coeff.comp")),
                ),
            ),
            pipeline_update_pressure_and_residual: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(
                    layout_pressure_update_pressure_and_residual.clone(),
                    &shader_path.join(&Path::new("pressure_update_pressure_and_residual.comp")),
                ),
            ),
            pipeline_update_search: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(
                    layout_pressure_update_volume.clone(),
                    &shader_path.join(&Path::new("pressure_update_search.comp")),
                ),
            ),

            volume_pressure_view: volume_pressure_from_velocity_view,
            volume_pcg_residual_view,

            is_first_step: Cell::new(true),
        }
    }

    pub fn pressure_view(&self) -> &wgpu::TextureView {
        &self.volume_pressure_view
    }

    pub fn residual_view(&self) -> &wgpu::TextureView {
        &self.volume_pcg_residual_view
    }

    fn reduce_add<'a, 'b: 'a>(&'b self, cpass: &mut wgpu::ComputePass<'a>, pipeline_manager: &'a PipelineManager, result_mode: u32) {
        // reduce.
        let mut source_buffer_index = 0;
        cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_reduce));
        let mut num_entries_remaining = (self.grid_dimension.width * self.grid_dimension.height * self.grid_dimension.depth) as u32;
        while num_entries_remaining > Self::DOTPRODUCT_REDUCE_REDUCTION_PER_STEP {
            cpass.set_bind_group(1, &self.bind_group_dotproduct_reduce[source_buffer_index], &[]);
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
        // final
        cpass.set_bind_group(1, &self.bind_group_dotproduct_final[source_buffer_index], &[]);
        cpass.set_push_constants(0, &[result_mode, num_entries_remaining]);
        cpass.dispatch(
            wgpu_utils::compute_group_size_1d(num_entries_remaining, Self::COMPUTE_LOCAL_SIZE_DOTPRODUCT_REDUCE),
            1,
            1,
        );
    }

    pub fn solve<'a, 'b: 'a>(&'b self, cpass: &mut wgpu::ComputePass<'a>, pipeline_manager: &'a PipelineManager) {
        let grid_work_groups = wgpu_utils::compute_group_size(self.grid_dimension, Self::COMPUTE_LOCAL_SIZE_VOLUME);

        cpass.set_bind_group(0, &self.bind_group_general, &[]);

        // For optimization various steps are collapsed as far as possible to avoid expensive buffer/texture read/writes
        // This makes the algorithm a lot faster but also a bit harder to read.

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
        if self.is_first_step.get() {
            cpass.set_push_constants(0, &[FIRST_STEP]);
        } else {
            cpass.set_push_constants(0, &[NOT_FIRST_STEP]);
        }
        cpass.set_bind_group(1, &self.bind_group_init, &[]);
        cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);

        // Apply preconditioner on (r), store result to search vector (s) and start dotproduct of <s; r>
        // Note that we don't use the auxillary vector here as in-between storage!
        const PRECONDITIONER_PASS0: u32 = 0;
        const PRECONDITIONER_PASS1: u32 = 1;
        const PRECONDITIONER_PASS1_SET_UNUSED_TO_ZERO: u32 = 3;
        cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_apply_preconditioner));
        cpass.set_push_constants(0, &[0]);
        cpass.set_push_constants(0, &[PRECONDITIONER_PASS0]);
        cpass.set_bind_group(1, &self.bind_group_preconditioner[0], &[]);
        cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);
        cpass.set_push_constants(0, &[PRECONDITIONER_PASS1_SET_UNUSED_TO_ZERO]);
        cpass.set_bind_group(1, &self.bind_group_preconditioner[2], &[]);
        cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);
        // Init sigma to dotproduct of search vector (s) and residual (r)
        self.reduce_add(cpass, pipeline_manager, Self::DOTPRODUCT_RESULTMODE_INIT);

        // Solver iterations ...
        const NUM_ITERATIONS: u32 = 16;
        let mut i = 0;
        loop {
            // Apply cell relationships to search vector (i.e. multiply s with A)
            // The dot product is applied to the result (denoted as z in Bridson's book) and the search vector (s), i.e. compute <s; As>
            cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_apply_coeff));
            cpass.set_bind_group(1, &self.bind_group_apply_coeff, &[]);
            cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);
            // finish dotproduct of auxiliary field (z) and search field (s)
            self.reduce_add(cpass, pipeline_manager, Self::DOTPRODUCT_RESULTMODE_ALPHA);

            // update pressure field (p) and residual field (r)
            const PRUPDATE_LAST_ITERATION: u32 = 1;
            cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_update_pressure_and_residual));
            if i == NUM_ITERATIONS {
                cpass.set_push_constants(0, &[PRUPDATE_LAST_ITERATION]);
            } else {
                cpass.set_push_constants(0, &[0]);
            }
            cpass.set_bind_group(1, &self.bind_group_update_pressure_and_residual, &[]);
            cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);
            if i == NUM_ITERATIONS {
                break;
            }

            // Apply preconditioner on (r), store result to auxillary (z) and start dotproduct of <z; r>
            cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_apply_preconditioner));
            cpass.set_push_constants(0, &[PRECONDITIONER_PASS0]);
            cpass.set_bind_group(1, &self.bind_group_preconditioner[0], &[]);
            cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);
            cpass.set_push_constants(0, &[PRECONDITIONER_PASS1]);
            cpass.set_bind_group(1, &self.bind_group_preconditioner[1], &[]);
            cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);
            // finish dotproduct of auxiliary field (z) and residual field (r)
            self.reduce_add(cpass, pipeline_manager, Self::DOTPRODUCT_RESULTMODE_BETA);

            // Update search vector
            cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_update_search));
            cpass.set_bind_group(1, &self.bind_group_update_search, &[]);
            cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);

            i += 1;
        }
        self.is_first_step.set(false);
    }
}
