use crate::wgpu_utils::binding_builder::*;
use crate::wgpu_utils::pipelines::*;
use crate::wgpu_utils::shader::*;
use crate::wgpu_utils::uniformbuffer::*;
use crate::wgpu_utils::*;
use rand::prelude::*;
use std::{cell::Cell, path::Path, rc::Rc};

#[repr(C)]
#[derive(Clone, Copy)]
struct SimulationPropertiesUniformBufferContent {
    gravity_grid: cgmath::Vector3<f32>,
    num_particles: u32,
}
unsafe impl bytemuck::Pod for SimulationPropertiesUniformBufferContent {}
unsafe impl bytemuck::Zeroable for SimulationPropertiesUniformBufferContent {}

#[repr(C)]
#[derive(Clone, Copy)]
struct TransferVelocityToGridUniformBufferContent {
    component: u32,
}
unsafe impl bytemuck::Pod for TransferVelocityToGridUniformBufferContent {}
unsafe impl bytemuck::Zeroable for TransferVelocityToGridUniformBufferContent {}

pub struct HybridFluid {
    grid_dimension: wgpu::Extent3d,

    particles_position_llindex: wgpu::Buffer,
    simulation_properties_uniformbuffer: UniformBuffer<SimulationPropertiesUniformBufferContent>,
    simulation_properties: SimulationPropertiesUniformBufferContent,
    simulation_properties_dirty: Cell<bool>,

    bind_group_uniform: wgpu::BindGroup,

    bind_group_transfer_velocity: [wgpu::BindGroup; 3],
    bind_group_write_velocity: wgpu::BindGroup,
    bind_group_write_particles: wgpu::BindGroup,
    bind_group_compute_divergence: wgpu::BindGroup,
    bind_group_pressure_write: [wgpu::BindGroup; 2],

    // The interface to any renderer of the fluid. Readonly access to relevant resources
    bind_group_renderer: wgpu::BindGroup,

    pipeline_transfer_clear_linkedlist: ComputePipelineHandle,
    pipeline_transfer_build_linkedlist: ComputePipelineHandle,
    pipeline_transfer_gather: ComputePipelineHandle,
    pipeline_compute_divergence: ComputePipelineHandle,
    pipeline_pressure_solve: ComputePipelineHandle,
    pipeline_remove_divergence: ComputePipelineHandle,
    pipeline_extrapolate_velocity: ComputePipelineHandle,
    pipeline_update_particles: ComputePipelineHandle,

    max_num_particles: u32,
}

static mut GROUP_LAYOUT_RENDERER: Option<BindGroupLayoutWithDesc> = None;

#[repr(C)]
#[derive(Clone, Copy)]
struct ParticlePositionLl {
    // Particle positions are in grid space to simplify shader computation
    // (no scaling/translation needed until we're rendering or interacting with other objects!)
    position: cgmath::Point3<f32>,
    linked_list_next: u32,
}
unsafe impl bytemuck::Pod for ParticlePositionLl {}
unsafe impl bytemuck::Zeroable for ParticlePositionLl {}

impl HybridFluid {
    // particles are distributed 2x2x2 within a single gridcell
    // (seems to be widely accepted as the default. Houdini seems to have this configurable from 4-16, maybe worth experimenting with it! (todo))
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
        let group_layout_transfer_velocity = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::buffer(false)) // particles, position llindex
            .next_binding_compute(binding_glsl::buffer(true)) // particles, velocity component
            .next_binding_compute(binding_glsl::uimage3d(wgpu::TextureFormat::R32Uint, false)) // linkedlist_volume
            .next_binding_compute(binding_glsl::uimage3d(wgpu::TextureFormat::R8Uint, false)) // marker volume
            .next_binding_compute(binding_glsl::image3d(wgpu::TextureFormat::R32Float, false)) // velocity component
            .next_binding_compute(binding_glsl::uniform())
            .create(device, "BindGroupLayout: Transfer velocity from Particles to Volume(s)");
        let group_layout_write_velocity = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::image3d(wgpu::TextureFormat::R32Float, false)) // velocityX
            .next_binding_compute(binding_glsl::image3d(wgpu::TextureFormat::R32Float, false)) // velocityY
            .next_binding_compute(binding_glsl::image3d(wgpu::TextureFormat::R32Float, false)) // velocityZ
            .next_binding_compute(binding_glsl::utexture3D()) // marker volume
            .next_binding_compute(binding_glsl::texture2D()) // pressure
            .create(device, "BindGroupLayout: Write to Velocity");
        let group_layout_write_particles = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::texture3D()) // velocityX
            .next_binding_compute(binding_glsl::texture3D()) // velocityY
            .next_binding_compute(binding_glsl::texture3D()) // velocityZ
            .next_binding_compute(binding_glsl::utexture3D()) // marker volume
            .next_binding_compute(binding_glsl::buffer(false)) // particles, position llindex
            .next_binding_compute(binding_glsl::buffer(false)) // particles, velocityX
            .next_binding_compute(binding_glsl::buffer(false)) // particles, velocityY
            .next_binding_compute(binding_glsl::buffer(false)) // particles, velocityZ
            .create(device, "BindGroupLayout: Write to Particles");
        let group_layout_pressure_solve = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::texture3D()) // velocityX
            .next_binding_compute(binding_glsl::texture3D()) // velocityY
            .next_binding_compute(binding_glsl::texture3D()) // velocityZ
            .next_binding_compute(binding_glsl::utexture3D()) // marker volume
            .next_binding_compute(binding_glsl::texture3D()) // dummy or divergence
            .next_binding_compute(binding_glsl::texture3D()) // pressure
            .next_binding_compute(binding_glsl::image3d(wgpu::TextureFormat::R32Float, false)) // pressure or divergence
            .create(device, "BindGroupLayout: Pressure solve volumes");

        // Resources
        let simulation_properties_uniformbuffer = UniformBuffer::new(device);
        let particles_position_llindex = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Buffer: Particles position & llindex"),
            size: max_num_particles as u64 * std::mem::size_of::<ParticlePositionLl>() as u64,
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
            mapped_at_creation: false,
        });
        let particles_velocity_x = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Buffer: Particles velocity X"),
            size: max_num_particles as u64 * std::mem::size_of::<cgmath::Vector4<f32>>() as u64,
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
            mapped_at_creation: false,
        });
        let particles_velocity_y = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Buffer: Particles velocity Y"),
            size: max_num_particles as u64 * std::mem::size_of::<cgmath::Vector4<f32>>() as u64,
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
            mapped_at_creation: false,
        });
        let particles_velocity_z = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Buffer: Particles velocity Z"),
            size: max_num_particles as u64 * std::mem::size_of::<cgmath::Vector4<f32>>() as u64,
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
            mapped_at_creation: false,
        });

        let create_volume_texture_descriptor = |label: &'static str, format: wgpu::TextureFormat| -> wgpu::TextureDescriptor {
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
        let volume_velocity_x = device.create_texture(&create_volume_texture_descriptor("Velocity Volume X", wgpu::TextureFormat::R32Float));
        let volume_velocity_y = device.create_texture(&create_volume_texture_descriptor("Velocity Volume Y", wgpu::TextureFormat::R32Float));
        let volume_velocity_z = device.create_texture(&create_volume_texture_descriptor("Velocity Volume Z", wgpu::TextureFormat::R32Float));
        let volume_linked_lists = device.create_texture(&create_volume_texture_descriptor("Linked Lists Volume", wgpu::TextureFormat::R32Uint));
        let volume_marker = device.create_texture(&create_volume_texture_descriptor("Marker Grid", wgpu::TextureFormat::R8Uint));
        let volume_divergence = device.create_texture(&create_volume_texture_descriptor("Velocity Volume", wgpu::TextureFormat::R32Float));
        let volume_pressure0 = device.create_texture(&create_volume_texture_descriptor("Pressure Volume 0", wgpu::TextureFormat::R32Float));
        let volume_pressure1 = device.create_texture(&create_volume_texture_descriptor("Pressure Volume 1", wgpu::TextureFormat::R32Float));
        let ubo_transfer_velocity = [
            UniformBuffer::new_with_data(device, &TransferVelocityToGridUniformBufferContent { component: 0 }),
            UniformBuffer::new_with_data(device, &TransferVelocityToGridUniformBufferContent { component: 1 }),
            UniformBuffer::new_with_data(device, &TransferVelocityToGridUniformBufferContent { component: 2 }),
        ];

        // Resource views
        let volume_velocity_view_x = volume_velocity_x.create_default_view();
        let volume_velocity_view_y = volume_velocity_y.create_default_view();
        let volume_velocity_view_z = volume_velocity_z.create_default_view();
        let volume_linked_lists_view = volume_linked_lists.create_default_view();
        let volume_marker_view = volume_marker.create_default_view();
        let volume_divergence_view = volume_divergence.create_default_view();
        let volume_pressure0_view = volume_pressure0.create_default_view();
        let volume_pressure1_view = volume_pressure1.create_default_view();

        // Bind groups.
        let bind_group_uniform = BindGroupBuilder::new(&group_layout_uniform)
            .resource(simulation_properties_uniformbuffer.binding_resource())
            .create(device, "BindGroup: HybridFluid Uniform");

        let bind_group_transfer_velocity = [
            BindGroupBuilder::new(&group_layout_transfer_velocity)
                .buffer(particles_position_llindex.slice(..))
                .buffer(particles_velocity_x.slice(..))
                .texture(&volume_linked_lists_view)
                .texture(&volume_marker_view)
                .texture(&volume_velocity_view_x)
                .resource(ubo_transfer_velocity[0].binding_resource())
                .create(device, "BindGroup: Transfer velocity to volume X"),
            BindGroupBuilder::new(&group_layout_transfer_velocity)
                .buffer(particles_position_llindex.slice(..))
                .buffer(particles_velocity_y.slice(..))
                .texture(&volume_linked_lists_view)
                .texture(&volume_marker_view)
                .texture(&volume_velocity_view_y)
                .resource(ubo_transfer_velocity[1].binding_resource())
                .create(device, "BindGroup: Transfer velocity to volume Y"),
            BindGroupBuilder::new(&group_layout_transfer_velocity)
                .buffer(particles_position_llindex.slice(..))
                .buffer(particles_velocity_z.slice(..))
                .texture(&volume_linked_lists_view)
                .texture(&volume_marker_view)
                .texture(&volume_velocity_view_z)
                .resource(ubo_transfer_velocity[2].binding_resource())
                .create(device, "BindGroup: Transfer velocity to volume Z"),
        ];

        let bind_group_write_velocity = BindGroupBuilder::new(&group_layout_write_velocity)
            .texture(&volume_velocity_view_x)
            .texture(&volume_velocity_view_y)
            .texture(&volume_velocity_view_z)
            .texture(&volume_marker_view)
            .texture(&volume_pressure0_view)
            .create(device, "BindGroup: Write to Velocity Grid");
        let bind_group_write_particles = BindGroupBuilder::new(&group_layout_write_particles)
            .texture(&volume_velocity_view_x)
            .texture(&volume_velocity_view_y)
            .texture(&volume_velocity_view_z)
            .texture(&volume_marker_view)
            .buffer(particles_position_llindex.slice(..))
            .buffer(particles_velocity_x.slice(..))
            .buffer(particles_velocity_y.slice(..))
            .buffer(particles_velocity_z.slice(..))
            .create(device, "BindGroup: Write to Particles");
        let bind_group_compute_divergence = BindGroupBuilder::new(&group_layout_pressure_solve)
            .texture(&volume_velocity_view_x)
            .texture(&volume_velocity_view_y)
            .texture(&volume_velocity_view_z)
            .texture(&volume_marker_view)
            .texture(&volume_pressure0_view)
            .texture(&volume_pressure1_view)
            .texture(&volume_divergence_view)
            .create(device, "BindGroup: Compute Divergence");
        let bind_group_pressure_write = [
            BindGroupBuilder::new(&group_layout_pressure_solve)
                .texture(&volume_velocity_view_x)
                .texture(&volume_velocity_view_y)
                .texture(&volume_velocity_view_z)
                .texture(&volume_marker_view)
                .texture(&volume_divergence_view)
                .texture(&volume_pressure1_view)
                .texture(&volume_pressure0_view)
                .create(device, "BindGroup: Pressure write 0"),
            BindGroupBuilder::new(&group_layout_pressure_solve)
                .texture(&volume_velocity_view_x)
                .texture(&volume_velocity_view_y)
                .texture(&volume_velocity_view_z)
                .texture(&volume_marker_view)
                .texture(&volume_divergence_view)
                .texture(&volume_pressure0_view)
                .texture(&volume_pressure1_view)
                .create(device, "BindGroup: Pressure write 1"),
        ];

        let bind_group_renderer = BindGroupBuilder::new(&Self::get_or_create_group_layout_renderer(device))
            .buffer(particles_position_llindex.slice(..))
            .buffer(particles_velocity_x.slice(..))
            .buffer(particles_velocity_y.slice(..))
            .buffer(particles_velocity_z.slice(..))
            .texture(&volume_velocity_view_x)
            .texture(&volume_velocity_view_y)
            .texture(&volume_velocity_view_z)
            .texture(&volume_marker_view)
            .texture(&volume_divergence_view)
            .texture(&volume_pressure0_view)
            .create(device, "BindGroup: Fluid Renderers");

        // pipeline layouts.
        // Note that layouts directly correspond to DX12 root signatures.
        // We want to avoid having many of them and share as much as we can, but since WebGPU needs to set barriers for everything that is not readonly it's a tricky tradeoff.
        // Considering that all pipelines here require UAV barriers anyways a few more or less won't make too much difference (... is that true?).
        // Therefore we're compromising for less layouts & easier to maintain code (also less binding changes 🤔)
        let layout_transfer_velocity = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[
                per_frame_bind_group_layout,
                &group_layout_uniform.layout,
                &group_layout_transfer_velocity.layout,
            ],
        }));
        let layout_write_volume = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[
                per_frame_bind_group_layout,
                &group_layout_uniform.layout,
                &group_layout_write_velocity.layout,
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

        let pipeline_transfer_clear_linkedlist = pipeline_manager.create_compute_pipeline(
            device,
            shader_dir,
            ComputePipelineCreationDesc::new(layout_transfer_velocity.clone(), Path::new("simulation/transfer_clear_linkedlist.comp")),
        );
        let pipeline_transfer_build_linkedlist = pipeline_manager.create_compute_pipeline(
            device,
            shader_dir,
            ComputePipelineCreationDesc::new(layout_transfer_velocity.clone(), Path::new("simulation/transfer_build_linkedlist.comp")),
        );
        let pipeline_transfer_gather = pipeline_manager.create_compute_pipeline(
            device,
            shader_dir,
            ComputePipelineCreationDesc::new(layout_transfer_velocity.clone(), Path::new("simulation/transfer_gather.comp")),
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
            ComputePipelineCreationDesc::new(layout_write_volume.clone(), Path::new("simulation/remove_divergence.comp")),
        );
        let pipeline_extrapolate_velocity = pipeline_manager.create_compute_pipeline(
            device,
            shader_dir,
            ComputePipelineCreationDesc::new(layout_write_volume.clone(), Path::new("simulation/extrapolate_velocity.comp")),
        );
        let pipeline_update_particles = pipeline_manager.create_compute_pipeline(
            device,
            shader_dir,
            ComputePipelineCreationDesc::new(layout_write_particles.clone(), Path::new("simulation/update_particles.comp")),
        );

        HybridFluid {
            grid_dimension,

            particles_position_llindex,
            simulation_properties_uniformbuffer,
            simulation_properties: SimulationPropertiesUniformBufferContent {
                num_particles: 0,
                gravity_grid: cgmath::vec3(0.0, -9.81, 0.0),
            },
            simulation_properties_dirty: Cell::new(true),

            bind_group_uniform,
            bind_group_transfer_velocity,
            bind_group_write_velocity,
            bind_group_write_particles,
            bind_group_compute_divergence,
            bind_group_pressure_write,

            bind_group_renderer,

            pipeline_transfer_clear_linkedlist,
            pipeline_transfer_build_linkedlist,
            pipeline_transfer_gather,
            pipeline_compute_divergence,
            pipeline_pressure_solve,
            pipeline_extrapolate_velocity,
            pipeline_remove_divergence,
            pipeline_update_particles,

            max_num_particles,
        }
    }

    fn clamp_to_grid(&self, grid_cor: cgmath::Point3<f32>) -> cgmath::Point3<u32> {
        // Due to the design of the grid, the 0-1 range is reserved by solid cells and can't be filled.
        cgmath::Point3::new(
            self.grid_dimension.width.min(grid_cor.x as u32).max(1),
            self.grid_dimension.height.min(grid_cor.y as u32).max(1),
            self.grid_dimension.depth.min(grid_cor.z as u32).max(1),
        )
    }

    // Adds a cube of fluid. Coordinates are in grid space! Very slow operation!
    pub fn add_fluid_cube(&mut self, queue: &wgpu::Queue, min_grid: cgmath::Point3<f32>, max_grid: cgmath::Point3<f32>) {
        // align to whole cells for simplicity.
        let min_grid = self.clamp_to_grid(min_grid);
        let max_grid = self.clamp_to_grid(max_grid);
        let extent_cell = max_grid - min_grid;

        let mut num_new_particles = (extent_cell.x * extent_cell.y * extent_cell.z * Self::PARTICLES_PER_GRID_CELL) as u32;
        if self.max_num_particles < num_new_particles + self.simulation_properties.num_particles {
            error!(
                "Can't add {} particles, max is {}, current is {}",
                num_new_particles, self.max_num_particles, self.simulation_properties.num_particles
            );
            num_new_particles = self.max_num_particles - self.simulation_properties.num_particles;
        }
        info!("Adding {} new particles", num_new_particles);

        // Fill buffer with particle data
        let mut rng: rand::rngs::SmallRng = rand::SeedableRng::seed_from_u64((self.simulation_properties.num_particles + num_new_particles) as u64);
        let mut new_particles = Vec::new();
        new_particles.resize(
            num_new_particles as usize,
            ParticlePositionLl {
                position: cgmath::point3(0.0, 0.0, 0.0),
                linked_list_next: 0xFFFFFFFF,
            },
        );
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

            particle.position = cell + offset;
        }

        let particle_size = std::mem::size_of::<ParticlePositionLl>() as u64;
        queue.write_buffer(
            &self.particles_position_llindex,
            self.simulation_properties.num_particles as u64 * particle_size,
            bytemuck::cast_slice(&new_particles),
        );
        self.simulation_properties.num_particles += num_new_particles;
        self.simulation_properties_dirty.set(true);
    }

    pub fn set_gravity_grid(&mut self, gravity: cgmath::Vector3<f32>) {
        self.simulation_properties.gravity_grid = gravity;
        self.simulation_properties_dirty.set(true);
    }

    pub fn num_particles(&self) -> u32 {
        self.simulation_properties.num_particles
    }

    pub fn get_or_create_group_layout_renderer(device: &wgpu::Device) -> &BindGroupLayoutWithDesc {
        unsafe {
            GROUP_LAYOUT_RENDERER.get_or_insert_with(|| {
                BindGroupLayoutBuilder::new()
                    .next_binding_vertex(binding_glsl::buffer(true)) // particles, position llindex
                    .next_binding_vertex(binding_glsl::buffer(true)) // particles, velocityX
                    .next_binding_vertex(binding_glsl::buffer(true)) // particles, velocityY
                    .next_binding_vertex(binding_glsl::buffer(true)) // particles, velocityZ
                    .next_binding_vertex(binding_glsl::texture3D()) // velocityX
                    .next_binding_vertex(binding_glsl::texture3D()) // velocityY
                    .next_binding_vertex(binding_glsl::texture3D()) // velocityZ
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

    pub fn step<'a>(&'a self, cpass: &mut wgpu::ComputePass<'a>, pipeline_manager: &'a PipelineManager, queue: &wgpu::Queue) {
        if self.simulation_properties_dirty.get() {
            self.simulation_properties_uniformbuffer.update_content(queue, self.simulation_properties);
            self.simulation_properties_dirty.set(false);
        }

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
        let particle_work_groups =
            (self.simulation_properties.num_particles as u32 + COMPUTE_LOCAL_SIZE_PARTICLES - 1) / COMPUTE_LOCAL_SIZE_PARTICLES;

        cpass.set_bind_group(1, &self.bind_group_uniform, &[]);

        // grouped by layouts.
        {
            for i in 0..3 {
                cpass.set_bind_group(2, &self.bind_group_transfer_velocity[i], &[]);

                // clear front velocity and linkedlist grid
                // It's either this or a loop over encoder.begin_render_pass which then also requires a myriad of texture views...
                // (might still be faster because RT clear operations are usually very quick :/)
                cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_transfer_clear_linkedlist));
                cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);

                // Create particle linked lists and write heads in dual grids
                // Transfer velocities to grid. (write grid, read particles)
                cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_transfer_build_linkedlist));
                cpass.dispatch(particle_work_groups, 1, 1);

                // Gather velocities in velocity grid and apply global forces.
                cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_transfer_gather));
                cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);
            }
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
            cpass.set_bind_group(2, &self.bind_group_write_velocity, &[]);

            // Make velocity grid divergence free
            cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_remove_divergence));
            cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);

            // Extrapolate velocity
            // can only do a single extrapolation since we can't change cell types without double buffering
            // (this makes the extrapolation a bit heavier since it needs to sample all 8 diagonals as well)
            cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_extrapolate_velocity));
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
