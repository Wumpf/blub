use crate::wgpu_utils::binding_builder::*;
use crate::wgpu_utils::pipelines::*;
use crate::wgpu_utils::shader::*;
use crate::wgpu_utils::uniformbuffer::*;
use crate::wgpu_utils::*;
use rand::prelude::*;
use std::{path::Path, rc::Rc};

#[repr(C)]
#[derive(Clone, Copy)]
struct SimulationPropertiesUniformBufferContent {
    num_particles: u32,
    padding0: f32,
    padding1: f32,
    padding2: f32,
}

pub struct HybridFluid {
    //gravity: cgmath::Vector3<f32>, // global gravity force in m/s² (== N/kg)
    grid_dimension: wgpu::Extent3d,

    particles: wgpu::Buffer,
    simulation_properties_uniformbuffer: UniformBuffer<SimulationPropertiesUniformBufferContent>,

    bind_group_uniform: wgpu::BindGroup,
    bind_group_write_particles_volume: wgpu::BindGroup,
    bind_group_write_particles: wgpu::BindGroup,
    bind_group_compute_divergence: wgpu::BindGroup,
    bind_group_pressure_write: [wgpu::BindGroup; 2],

    // The interface to any renderer of the fluid. Readonly access to relevant resources
    bind_group_renderer: wgpu::BindGroup,

    pipeline_clear_llgrid: ComputePipelineHandle,
    pipeline_build_linkedlist_volume: ComputePipelineHandle,
    pipeline_transfer_to_volume: ComputePipelineHandle,
    pipeline_compute_divergence: ComputePipelineHandle,
    pipeline_pressure_solve: ComputePipelineHandle,
    pipeline_remove_divergence: ComputePipelineHandle,
    pipeline_update_particles: ComputePipelineHandle,

    num_particles: u32,
    max_num_particles: u32,
}

static mut GROUP_LAYOUT_RENDERER: Option<BindGroupLayoutWithDesc> = None;

// todo: probably want to split this up into several buffers
#[repr(C)]
#[derive(Clone, Copy)]
struct Particle {
    // Particle positions are in grid space to simplify shader computation
    // (no scaling/translation needed until we're rendering or interacting with other objects!)
    position: cgmath::Point3<f32>,
    linked_list_next: u32,

    velocity_matrix_0: cgmath::Vector4<f32>,
    velocity_matrix_1: cgmath::Vector4<f32>,
    velocity_matrix_2: cgmath::Vector4<f32>,
}

impl HybridFluid {
    // particles are distributed 2x2x2 within a single gridcell
    // (seems to be widely accepted as the default)
    const PARTICLES_PER_GRID_CELL: u32 = 8;

    pub fn new(
        device: &wgpu::Device,
        grid_dimension: wgpu::Extent3d,
        max_num_particles: u32,
        shader_dir: &ShaderDirectory,
        pipeline_manager: &mut PipelineManager,
        per_frame_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        // Layouts
        let group_layout_uniform = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::uniform())
            .create(device, "BindGroupLayout: HybridFluid Uniform");
        let group_layout_write_particles_volume = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::buffer(false)) // particles
            .next_binding_compute(binding_glsl::image3d(wgpu::TextureFormat::Rgba32Float, false)) // vgrid
            .next_binding_compute(binding_glsl::uimage3d(wgpu::TextureFormat::R32Uint, false)) // linkedlist_volume
            .next_binding_compute(binding_glsl::uimage3d(wgpu::TextureFormat::R8Uint, false)) // marker volume
            .next_binding_compute(binding_glsl::texture2D()) // pressure
            .create(device, "BindGroupLayout: Update Particles and/or Velocity Grid");
        let group_layout_write_particles = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::buffer(false)) // particles
            .next_binding_compute(binding_glsl::texture3D()) // vgrid
            .next_binding_compute(binding_glsl::utexture3D()) // marker volume
            .create(device, "BindGroupLayout: Update Particles and/or Velocity Grid");
        let group_layout_pressure_solve = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::texture3D()) // vgrid
            .next_binding_compute(binding_glsl::utexture3D()) // marker volume
            .next_binding_compute(binding_glsl::texture3D()) // dummy or divergence
            .next_binding_compute(binding_glsl::texture3D()) // pressure
            .next_binding_compute(binding_glsl::image3d(wgpu::TextureFormat::R32Float, false)) // pressure or divergence
            .create(device, "BindGroupLayout: Pressure solve volumes");

        // Resources
        let simulation_properties_uniformbuffer = UniformBuffer::new(device);
        let particle_buffer_size = max_num_particles as u64 * std::mem::size_of::<Particle>() as u64;
        let particles = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Buffer: ParticleBuffer"),
            size: particle_buffer_size,
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::STORAGE_READ | wgpu::BufferUsage::COPY_DST,
        });
        let create_volume_texture_descriptor = |label: &'static str, format: wgpu::TextureFormat| -> wgpu::TextureDescriptor {
            wgpu::TextureDescriptor {
                label: Some(label),
                size: grid_dimension,
                array_layer_count: 1,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format,
                usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::STORAGE,
            }
        };
        let volume_velocity = device.create_texture(&create_volume_texture_descriptor("Velocity Volume", wgpu::TextureFormat::Rgba32Float));
        let volume_linked_lists = device.create_texture(&create_volume_texture_descriptor("Linked Lists Volume", wgpu::TextureFormat::R32Uint));
        let volume_marker = device.create_texture(&create_volume_texture_descriptor("Marker Grid", wgpu::TextureFormat::R8Uint));
        let volume_divergence = device.create_texture(&create_volume_texture_descriptor("Velocity Volume", wgpu::TextureFormat::R32Float)); // TODO: could reuse data from volume_linked_lists
        let volume_pressure0 = device.create_texture(&create_volume_texture_descriptor("Pressure Volume 0", wgpu::TextureFormat::R32Float));
        let volume_pressure1 = device.create_texture(&create_volume_texture_descriptor("Pressure Volume 1", wgpu::TextureFormat::R32Float));

        // Resource views
        let volume_velocity_view = volume_velocity.create_default_view();
        let volume_linked_lists_view = volume_linked_lists.create_default_view();
        let volume_marker_view = volume_marker.create_default_view();
        let volume_divergence_view = volume_divergence.create_default_view();
        let volume_pressure0_view = volume_pressure0.create_default_view();
        let volume_pressure1_view = volume_pressure1.create_default_view();

        // Bind groups.
        let bind_group_uniform = BindGroupBuilder::new(&group_layout_uniform)
            .resource(simulation_properties_uniformbuffer.binding_resource())
            .create(device, "BindGroup: HybridFluid Uniform");
        let bind_group_write_particles_volume = BindGroupBuilder::new(&group_layout_write_particles_volume)
            .buffer(&particles, 0..particle_buffer_size)
            .texture(&volume_velocity_view)
            .texture(&volume_linked_lists_view)
            .texture(&volume_marker_view)
            .texture(&volume_pressure0_view)
            .create(device, "BindGroup: Update Particles and/or Velocity Grid");
        let bind_group_write_particles = BindGroupBuilder::new(&group_layout_write_particles)
            .buffer(&particles, 0..particle_buffer_size)
            .texture(&volume_velocity_view)
            .texture(&volume_marker_view)
            .create(device, "BindGroup: Update Particles");
        let bind_group_compute_divergence = BindGroupBuilder::new(&group_layout_pressure_solve)
            .texture(&volume_velocity_view)
            .texture(&volume_marker_view)
            .texture(&volume_pressure0_view)
            .texture(&volume_pressure1_view)
            .texture(&volume_divergence_view)
            .create(device, "BindGroup: Compute Divergence");
        let bind_group_pressure_write = [
            BindGroupBuilder::new(&group_layout_pressure_solve)
                .texture(&volume_velocity_view)
                .texture(&volume_marker_view)
                .texture(&volume_divergence_view)
                .texture(&volume_pressure1_view)
                .texture(&volume_pressure0_view)
                .create(device, "BindGroup: Pressure write 0"),
            BindGroupBuilder::new(&group_layout_pressure_solve)
                .texture(&volume_velocity_view)
                .texture(&volume_marker_view)
                .texture(&volume_divergence_view)
                .texture(&volume_pressure0_view)
                .texture(&volume_pressure1_view)
                .create(device, "BindGroup: Pressure write 1"),
        ];

        let bind_group_renderer = BindGroupBuilder::new(&Self::get_or_create_group_layout_renderer(device))
            .buffer(&particles, 0..particle_buffer_size)
            .texture(&volume_velocity_view)
            .texture(&volume_marker_view)
            .texture(&volume_divergence_view)
            .texture(&volume_pressure0_view)
            .create(device, "BindGroup: Fluid Renderers");

        // pipeline layouts.
        // Note that layouts directly correspond to DX12 root signatures.
        // We want to avoid having many of them and share as much as we can, but since WebGPU needs to set barriers for everything that is not readonly it's a tricky tradeoff.
        // Considering that all pipelines here require UAV barriers anyways a few more or less won't make too much difference (... is that true?).
        // Therefore we're compromising for less layouts & easier to maintain code (also less binding changes 🤔)
        // TODO: This setup is super coarse now. Need to figure out actual impact and see if splitting down makes sense.
        let layout_write_particles_volume = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[
                per_frame_bind_group_layout,
                &group_layout_uniform.layout,
                &group_layout_write_particles_volume.layout,
            ],
        }));
        let layout_pressure_solve = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[
                per_frame_bind_group_layout,
                &group_layout_uniform.layout,
                &group_layout_pressure_solve.layout,
            ],
        }));
        let layout_write_particles = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[
                per_frame_bind_group_layout,
                &group_layout_uniform.layout,
                &group_layout_write_particles.layout,
            ],
        }));

        let pipeline_clear_llgrid = pipeline_manager.create_compute_pipeline(
            device,
            shader_dir,
            ComputePipelineCreationDesc::new(layout_write_particles_volume.clone(), Path::new("simulation/clear_llgrid.comp")),
        );
        let pipeline_build_linkedlist_volume = pipeline_manager.create_compute_pipeline(
            device,
            shader_dir,
            ComputePipelineCreationDesc::new(
                layout_write_particles_volume.clone(),
                Path::new("simulation/build_linkedlist_volume.comp"),
            ),
        );
        let pipeline_transfer_to_volume = pipeline_manager.create_compute_pipeline(
            device,
            shader_dir,
            ComputePipelineCreationDesc::new(layout_write_particles_volume.clone(), Path::new("simulation/transfer_to_volume.comp")),
        );
        let pipeline_compute_divergence = pipeline_manager.create_compute_pipeline(
            device,
            shader_dir,
            ComputePipelineCreationDesc::new(layout_pressure_solve.clone(), Path::new("simulation/compute_divergence.comp")),
        );
        let pipeline_pressure_solve = pipeline_manager.create_compute_pipeline(
            device,
            shader_dir,
            ComputePipelineCreationDesc::new(layout_pressure_solve.clone(), Path::new("simulation/pressure_solve.comp")),
        );
        let pipeline_remove_divergence = pipeline_manager.create_compute_pipeline(
            device,
            shader_dir,
            ComputePipelineCreationDesc::new(layout_write_particles_volume.clone(), Path::new("simulation/remove_divergence.comp")),
        );
        let pipeline_update_particles = pipeline_manager.create_compute_pipeline(
            device,
            shader_dir,
            ComputePipelineCreationDesc::new(layout_write_particles.clone(), Path::new("simulation/update_particles.comp")),
        );

        HybridFluid {
            //gravity: cgmath::Vector3::new(0.0, -9.81, 0.0), // there needs to be some grid->world relation
            grid_dimension,

            particles,
            simulation_properties_uniformbuffer,

            bind_group_uniform,
            bind_group_write_particles_volume,
            bind_group_write_particles,
            bind_group_compute_divergence,
            bind_group_pressure_write,

            bind_group_renderer,

            pipeline_clear_llgrid,
            pipeline_build_linkedlist_volume,
            pipeline_transfer_to_volume,
            pipeline_compute_divergence,
            pipeline_pressure_solve,
            pipeline_remove_divergence,
            pipeline_update_particles,

            num_particles: 0,
            max_num_particles,
        }
    }

    fn clamp_to_grid(&self, grid_cor: cgmath::Point3<f32>) -> cgmath::Point3<u32> {
        cgmath::Point3::new(
            self.grid_dimension.width.min(grid_cor.x as u32),
            self.grid_dimension.height.min(grid_cor.y as u32),
            self.grid_dimension.depth.min(grid_cor.z as u32),
        )
    }

    // Adds a cube of fluid. Coordinates are in grid space! Very slow operation!
    pub fn add_fluid_cube(
        &mut self,
        device: &wgpu::Device,
        init_encoder: &mut wgpu::CommandEncoder,
        min_grid: cgmath::Point3<f32>,
        max_grid: cgmath::Point3<f32>,
    ) {
        // align to whole cells for simplicity.
        let min_grid = self.clamp_to_grid(min_grid);
        let max_grid = self.clamp_to_grid(max_grid);
        let extent_cell = max_grid - min_grid;

        let mut num_new_particles = (extent_cell.x * extent_cell.y * extent_cell.z * Self::PARTICLES_PER_GRID_CELL) as u32;
        if self.max_num_particles < num_new_particles + self.num_particles {
            error!(
                "Can't add {} particles, max is {}, current is {}",
                num_new_particles, self.max_num_particles, self.num_particles
            );
            num_new_particles = self.max_num_particles - self.num_particles;
        }
        info!("Adding {} new particles", num_new_particles);

        let particle_size = std::mem::size_of::<Particle>() as u64;
        let particle_buffer_mapping = device.create_buffer_mapped(&wgpu::BufferDescriptor {
            label: Some("Buffer: Particle Update"),
            size: num_new_particles as u64 * particle_size,
            usage: wgpu::BufferUsage::COPY_SRC,
        });

        // Fill buffer with particle data
        let mut rng: rand::rngs::SmallRng = rand::SeedableRng::seed_from_u64(num_new_particles as u64);
        let new_particles =
            unsafe { std::slice::from_raw_parts_mut(particle_buffer_mapping.data.as_mut_ptr() as *mut Particle, num_new_particles as usize) };
        for (i, particle) in new_particles.iter_mut().enumerate() {
            let cell = cgmath::point3(
                (min_grid.x + i as u32 / Self::PARTICLES_PER_GRID_CELL % extent_cell.x) as f32,
                (min_grid.y + i as u32 / Self::PARTICLES_PER_GRID_CELL / extent_cell.x % extent_cell.y) as f32,
                (min_grid.z + i as u32 / Self::PARTICLES_PER_GRID_CELL / extent_cell.x / extent_cell.y) as f32,
            );

            let sample_idx = i as u32 % Self::PARTICLES_PER_GRID_CELL;

            // pure random
            // let offset = rng.gen::<cgmath::Vector3<f32>>();

            // pure regular
            // let offset = cgmath::vec3(
            //     (sample_idx % 2) as f32 + 0.5,
            //     (sample_idx / 2 % 2) as f32 + 0.5,
            //     (sample_idx / 4 % 2) as f32 + 0.5,
            // ) * 0.5;

            // stratified
            let offset = cgmath::vec3((sample_idx % 2) as f32, (sample_idx / 2 % 2) as f32, (sample_idx / 4 % 2) as f32) * 0.5
                + rng.gen::<cgmath::Vector3<f32>>() * 0.5;

            let position = cell + offset;

            *particle = Particle {
                position,
                linked_list_next: 0xFFFFFFFF,
                velocity_matrix_0: cgmath::Zero::zero(),
                velocity_matrix_1: cgmath::Zero::zero(),
                velocity_matrix_2: cgmath::Zero::zero(),
            };
        }

        init_encoder.copy_buffer_to_buffer(
            &particle_buffer_mapping.finish(),
            0,
            &self.particles,
            self.num_particles as u64 * particle_size,
            num_new_particles as u64 * particle_size,
        );
        self.num_particles += num_new_particles;

        self.update_simulation_properties_uniformbuffer(device, init_encoder);
    }

    fn update_simulation_properties_uniformbuffer(&mut self, device: &wgpu::Device, init_encoder: &mut wgpu::CommandEncoder) {
        self.simulation_properties_uniformbuffer.update_content(
            init_encoder,
            device,
            SimulationPropertiesUniformBufferContent {
                num_particles: self.num_particles,
                padding0: 0.0,
                padding1: 0.0,
                padding2: 0.0,
            },
        );
    }

    pub fn num_particles(&self) -> u32 {
        self.num_particles
    }

    pub fn get_or_create_group_layout_renderer(device: &wgpu::Device) -> &BindGroupLayoutWithDesc {
        unsafe {
            GROUP_LAYOUT_RENDERER.get_or_insert_with(|| {
                BindGroupLayoutBuilder::new()
                    .next_binding_vertex(binding_glsl::buffer(true)) // particles
                    .next_binding_vertex(binding_glsl::texture3D()) // velocity
                    .next_binding_vertex(binding_glsl::utexture3D()) // marker
                    .next_binding_vertex(binding_glsl::texture3D()) // divergence
                    .next_binding_vertex(binding_glsl::texture3D()) // pressure
                    .create(device, "BindGroupLayout: ParticleRenderer")
            })
        }
    }

    pub fn bind_group_renderer(&self) -> &wgpu::BindGroup {
        &self.bind_group_renderer
    }

    pub fn grid_dimension(&self) -> wgpu::Extent3d {
        self.grid_dimension
    }

    // todo: timing
    pub fn step<'a>(&'a self, cpass: &mut wgpu::ComputePass<'a>, pipeline_manager: &'a PipelineManager) {
        const COMPUTE_LOCAL_SIZE_FLUID: wgpu::Extent3d = wgpu::Extent3d {
            width: 8,
            height: 8,
            depth: 8,
        };
        const COMPUTE_LOCAL_SIZE_PARTICLES: u32 = 512;

        let grid_work_groups = wgpu::Extent3d {
            width: self.grid_dimension.width / COMPUTE_LOCAL_SIZE_FLUID.width,
            height: self.grid_dimension.height / COMPUTE_LOCAL_SIZE_FLUID.height,
            depth: self.grid_dimension.depth / COMPUTE_LOCAL_SIZE_FLUID.depth,
        };
        let particle_work_groups = (self.num_particles as u32 + COMPUTE_LOCAL_SIZE_PARTICLES - 1) / COMPUTE_LOCAL_SIZE_PARTICLES;

        cpass.set_bind_group(1, &self.bind_group_uniform, &[]);

        // grouped by layouts.
        {
            cpass.set_bind_group(2, &self.bind_group_write_particles_volume, &[]);

            // clear front velocity and linkedlist grid
            // It's either this or a loop over encoder.begin_render_pass which then also requires a myriad of texture views...
            // (might still be faster because RT clear operations are usually very quick :/)
            cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_clear_llgrid));
            cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);

            // Create particle linked lists and write heads in dual grids
            // Transfer velocities to grid. (write grid, read particles)
            cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_build_linkedlist_volume));
            cpass.dispatch(particle_work_groups, 1, 1);

            // Gather velocities in velocity grid and apply global forces.
            cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_transfer_to_volume));
            cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);
        }
        {
            // Compute divergence
            cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_compute_divergence));
            cpass.set_bind_group(2, &self.bind_group_compute_divergence, &[]);
            cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);

            // Pressure solve (last step needs to write to target 0, because that's what we read later again)
            // We reuse the pressure values from last time as initial guess. Since we run the simulation quite frequently, we don't need a lot of steps.
            // TODO: how many? Need to measure remaining divergence to make any meaningful statement about this.
            cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_pressure_solve));
            for i in 0..32 {
                cpass.set_bind_group(2, &self.bind_group_pressure_write[(i + 1) % 2], &[]);
                cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);
            }
        }
        {
            cpass.set_bind_group(2, &self.bind_group_write_particles_volume, &[]);

            // Make velocity grid divergence free
            cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_remove_divergence));
            cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);
        }
        {
            cpass.set_bind_group(2, &self.bind_group_write_particles, &[]);

            // Transfer velocities to particles.
            cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_update_particles));
            cpass.dispatch(particle_work_groups, 1, 1);
        }
    }
}
