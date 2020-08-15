use super::pressure_solver::PressureSolver;
use crate::wgpu_utils;
use crate::wgpu_utils::binding_builder::*;
use crate::wgpu_utils::binding_glsl;
use crate::wgpu_utils::pipelines::*;
use crate::wgpu_utils::shader::*;
use crate::wgpu_utils::uniformbuffer::*;
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

pub struct HybridFluid {
    grid_dimension: wgpu::Extent3d,

    pressure_solver: PressureSolver,

    particles_position_llindex: wgpu::Buffer,
    particles_velocity_x: wgpu::Buffer,
    particles_velocity_y: wgpu::Buffer,
    particles_velocity_z: wgpu::Buffer,
    simulation_properties_uniformbuffer: UniformBuffer<SimulationPropertiesUniformBufferContent>,
    simulation_properties: SimulationPropertiesUniformBufferContent,
    simulation_properties_dirty: Cell<bool>,

    bind_group_uniform: wgpu::BindGroup,
    bind_group_transfer_velocity: [wgpu::BindGroup; 3],
    bind_group_divergence_compute: wgpu::BindGroup,
    bind_group_write_velocity: wgpu::BindGroup,
    bind_group_read_mac_grid: wgpu::BindGroup,
    bind_group_advect_particles: wgpu::BindGroup,
    bind_group_density_projection_gather_error: wgpu::BindGroup,

    // The interface to any renderer of the fluid. Readonly access to relevant resources
    bind_group_renderer: wgpu::BindGroup,

    pipeline_transfer_clear: ComputePipelineHandle,
    pipeline_transfer_build_linkedlist: ComputePipelineHandle,
    pipeline_transfer_gather_velocity: ComputePipelineHandle,
    pipeline_divergence_compute: ComputePipelineHandle,
    pipeline_divergence_remove: ComputePipelineHandle,
    pipeline_extrapolate_velocity: ComputePipelineHandle,
    pipeline_advect_particles: ComputePipelineHandle,
    pipeline_density_projection_gather_error: ComputePipelineHandle,

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
    // (seems to be widely accepted as the default. Houdini seems to have this configurable from 4-16, maybe worth experimenting with it! Note however, that the density error computation assumes this constant as well!)
    pub const PARTICLES_PER_GRID_CELL: u32 = 8;

    pub fn new(
        device: &wgpu::Device,
        grid_dimension: wgpu::Extent3d,
        max_num_particles: u32,
        shader_dir: &ShaderDirectory,
        pipeline_manager: &mut PipelineManager,
        per_frame_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
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

        // TODO:
        // Various sources, old and new, claim that on Nvidia hardware 3D textures are actually 2d slices!
        // http://www-ppl.ist.osaka-u.ac.jp/research/papers/201405_sugimoto_pc.pdf
        // https://www.sciencedirect.com/science/article/pii/S2468502X1730027X#fig1
        // https://forum.unity.com/threads/improving-performance-of-3d-textures-using-texture-arrays.725384/#post-4849571
        // For Intel this is directly documented
        // https://www.x.org/docs/intel/BYT/intel_os_gfx_prm_vol5_-_memory_views.pdf
        // Wasn't able to find anything on AMD, but there is sources implying the layered nature
        // ("Two bilinear fetches are required when sampling from a volume texture with bilinear
        // filtering.")[http://amd-dev.wpengine.netdna-cdn.com/wordpress/media/2013/05/GCNPerformanceTweets.pdf]

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
        // TODO: Reuse volumes to safe memory, not all are used simultaneously.
        let volume_velocity_x = device.create_texture(&create_volume_texture_desc("Velocity Volume X", wgpu::TextureFormat::R32Float));
        let volume_velocity_y = device.create_texture(&create_volume_texture_desc("Velocity Volume Y", wgpu::TextureFormat::R32Float));
        let volume_velocity_z = device.create_texture(&create_volume_texture_desc("Velocity Volume Z", wgpu::TextureFormat::R32Float));
        let volume_linked_lists = device.create_texture(&create_volume_texture_desc("Linked Lists Volume", wgpu::TextureFormat::R32Uint));
        let volume_marker = device.create_texture(&create_volume_texture_desc("Marker Grid", wgpu::TextureFormat::R8Snorm));

        let volume_density = device.create_texture(&create_volume_texture_desc("Density Volume", wgpu::TextureFormat::R32Float));

        // Resource views
        let volume_velocity_view_x = volume_velocity_x.create_view(&Default::default());
        let volume_velocity_view_y = volume_velocity_y.create_view(&Default::default());
        let volume_velocity_view_z = volume_velocity_z.create_view(&Default::default());
        let volume_linked_lists_view = volume_linked_lists.create_view(&Default::default());
        let volume_marker_view = volume_marker.create_view(&Default::default());
        let volume_density_view = volume_density.create_view(&Default::default());

        // Layouts
        let group_layout_uniform = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::uniform())
            .create(device, "BindGroupLayout: HybridFluid Uniform");
        let group_layout_transfer_velocity = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::buffer(false)) // particles, position llindex
            .next_binding_compute(binding_glsl::buffer(true)) // particles, velocity component
            .next_binding_compute(binding_glsl::uimage3d(wgpu::TextureFormat::R32Uint, false)) // linkedlist_volume
            .next_binding_compute(binding_glsl::image3d(wgpu::TextureFormat::R8Snorm, false)) // marker volume
            .next_binding_compute(binding_glsl::image3d(wgpu::TextureFormat::R32Float, false)) // velocity component
            .create(device, "BindGroupLayout: Transfer velocity from Particles to Volume(s)");
        let group_layout_divergence_compute = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::texture3D()) // marker volume
            .next_binding_compute(binding_glsl::texture3D()) // velocityX
            .next_binding_compute(binding_glsl::texture3D()) // velocityY
            .next_binding_compute(binding_glsl::texture3D()) // velocityZ
            .next_binding_compute(binding_glsl::image3d(wgpu::TextureFormat::R32Float, false)) // divergence / initial residual
            .create(device, "BindGroupLayout: Compute Divergence");
        let group_layout_write_velocity_volume = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::image3d(wgpu::TextureFormat::R32Float, false)) // velocityX
            .next_binding_compute(binding_glsl::image3d(wgpu::TextureFormat::R32Float, false)) // velocityY
            .next_binding_compute(binding_glsl::image3d(wgpu::TextureFormat::R32Float, false)) // velocityZ
            .next_binding_compute(binding_glsl::texture3D()) // marker volume
            .next_binding_compute(binding_glsl::texture2D()) // pressure
            .create(device, "BindGroupLayout: Write to Velocity");
        let group_layout_advect_particles = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::texture2D()) // velocityX
            .next_binding_compute(binding_glsl::texture2D()) // velocityY
            .next_binding_compute(binding_glsl::texture2D()) // velocityZ
            .next_binding_compute(binding_glsl::image3d(wgpu::TextureFormat::R8Snorm, false)) // marker volume
            .next_binding_compute(binding_glsl::uimage3d(wgpu::TextureFormat::R32Uint, false)) // linkedlist_volume
            .next_binding_compute(binding_glsl::buffer(false)) // particles, position llindex
            .next_binding_compute(binding_glsl::buffer(false)) // particles, velocityX
            .next_binding_compute(binding_glsl::buffer(false)) // particles, velocityY
            .next_binding_compute(binding_glsl::buffer(false)) // particles, velocityZ
            .create(device, "BindGroupLayout: Advect to Particles");
        let group_layout_read_macgrid = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::texture3D()) // velocityX
            .next_binding_compute(binding_glsl::texture3D()) // velocityY
            .next_binding_compute(binding_glsl::texture3D()) // velocityZ
            .next_binding_compute(binding_glsl::texture3D()) // marker volume
            .create(device, "BindGroupLayout: Read MAC Grid");

        let group_layout_density_projection_gather_error = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::buffer(false)) // particles, position llindex
            .next_binding_compute(binding_glsl::utexture3D()) // linkedlist_volume
            .next_binding_compute(binding_glsl::image3d(wgpu::TextureFormat::R8Snorm, false)) // marker volume
            .next_binding_compute(binding_glsl::image3d(wgpu::TextureFormat::R32Float, false)) // density volume
            .create(device, "BindGroupLayout: Transfer velocity from Particles to Volume(s)");

        let pressure_solver = PressureSolver::new(
            device,
            grid_dimension,
            shader_dir,
            pipeline_manager,
            per_frame_bind_group_layout,
            &volume_marker_view,
        );

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
                .create(device, "BindGroup: Transfer velocity to volume X"),
            BindGroupBuilder::new(&group_layout_transfer_velocity)
                .buffer(particles_position_llindex.slice(..))
                .buffer(particles_velocity_y.slice(..))
                .texture(&volume_linked_lists_view)
                .texture(&volume_marker_view)
                .texture(&volume_velocity_view_y)
                .create(device, "BindGroup: Transfer velocity to volume Y"),
            BindGroupBuilder::new(&group_layout_transfer_velocity)
                .buffer(particles_position_llindex.slice(..))
                .buffer(particles_velocity_z.slice(..))
                .texture(&volume_linked_lists_view)
                .texture(&volume_marker_view)
                .texture(&volume_velocity_view_z)
                .create(device, "BindGroup: Transfer velocity to volume Z"),
        ];
        let bind_group_divergence_compute = BindGroupBuilder::new(&group_layout_divergence_compute)
            .texture(&volume_marker_view)
            .texture(&volume_velocity_view_x)
            .texture(&volume_velocity_view_y)
            .texture(&volume_velocity_view_z)
            .texture(pressure_solver.residual_view())
            .create(device, "BindGroup: Compute divergence");
        let bind_group_write_velocity = BindGroupBuilder::new(&group_layout_write_velocity_volume)
            .texture(&volume_velocity_view_x)
            .texture(&volume_velocity_view_y)
            .texture(&volume_velocity_view_z)
            .texture(&volume_marker_view)
            .texture(pressure_solver.pressure_view())
            .create(device, "BindGroup: Write to Velocity Grid");
        let bind_group_advect_particles = BindGroupBuilder::new(&group_layout_advect_particles)
            .texture(&volume_velocity_view_x)
            .texture(&volume_velocity_view_y)
            .texture(&volume_velocity_view_z)
            .texture(&volume_marker_view)
            .texture(&volume_linked_lists_view)
            .buffer(particles_position_llindex.slice(..))
            .buffer(particles_velocity_x.slice(..))
            .buffer(particles_velocity_y.slice(..))
            .buffer(particles_velocity_z.slice(..))
            .create(device, "BindGroup: Write to Particles");
        let bind_group_read_mac_grid = BindGroupBuilder::new(&group_layout_read_macgrid)
            .texture(&volume_velocity_view_x)
            .texture(&volume_velocity_view_y)
            .texture(&volume_velocity_view_z)
            .texture(&volume_marker_view)
            .create(device, "BindGroup: Read MAC Grid");
        let bind_group_density_projection_gather_error = BindGroupBuilder::new(&group_layout_density_projection_gather_error)
            .buffer(particles_position_llindex.slice(..))
            .texture(&volume_linked_lists_view)
            .texture(&volume_marker_view)
            .texture(&volume_density_view)
            .create(device, "BindGroup: Density projection gather");
        let bind_group_renderer = BindGroupBuilder::new(&Self::get_or_create_group_layout_renderer(device))
            .buffer(particles_position_llindex.slice(..))
            .buffer(particles_velocity_x.slice(..))
            .buffer(particles_velocity_y.slice(..))
            .buffer(particles_velocity_z.slice(..))
            .texture(&volume_velocity_view_x)
            .texture(&volume_velocity_view_y)
            .texture(&volume_velocity_view_z)
            .texture(&volume_marker_view)
            .texture(&pressure_solver.pressure_view())
            .texture(&volume_density_view)
            .create(device, "BindGroup: Fluid Renderers");

        // pipeline layouts.
        // Use same push constant range for all pipelines to improve internal Vulkan pipeline compatibility.
        let push_constant_ranges = &[wgpu::PushConstantRange {
            stages: wgpu::ShaderStage::COMPUTE,
            range: 0..8,
        }];

        let layout_transfer_velocity = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PipelineLayout: HybridFluid, Transfer Velocity"),
            bind_group_layouts: &[
                per_frame_bind_group_layout,
                &group_layout_uniform.layout,
                &group_layout_transfer_velocity.layout,
            ],
            push_constant_ranges,
        }));
        let layout_divergence_compute = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PipelineLayout: HybridFluid, Compute Divergence"),
            bind_group_layouts: &[per_frame_bind_group_layout, &group_layout_divergence_compute.layout],
            push_constant_ranges,
        }));
        let layout_write_velocity_volume = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PipelineLayout: HybridFluid, Write Volume"),
            bind_group_layouts: &[
                per_frame_bind_group_layout,
                &group_layout_uniform.layout,
                &group_layout_write_velocity_volume.layout,
            ],
            push_constant_ranges,
        }));
        let layout_particles = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PipelineLayout: HybridFluid, Particles"),
            bind_group_layouts: &[
                per_frame_bind_group_layout,
                &group_layout_uniform.layout,
                &group_layout_advect_particles.layout,
            ],
            push_constant_ranges,
        }));
        let layout_density_projection_gather_error = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PipelineLayout: HybridFluid, Density Projection Gather"),
            bind_group_layouts: &[
                per_frame_bind_group_layout,
                &group_layout_uniform.layout,
                &group_layout_density_projection_gather_error.layout,
            ],
            push_constant_ranges,
        }));

        HybridFluid {
            grid_dimension,

            pressure_solver,

            particles_position_llindex,
            particles_velocity_x,
            particles_velocity_y,
            particles_velocity_z,
            simulation_properties_uniformbuffer,
            simulation_properties: SimulationPropertiesUniformBufferContent {
                num_particles: 0,
                gravity_grid: cgmath::vec3(0.0, -9.81, 0.0),
            },
            simulation_properties_dirty: Cell::new(true),

            bind_group_uniform,
            bind_group_transfer_velocity,
            bind_group_divergence_compute,
            bind_group_write_velocity,
            bind_group_advect_particles,

            bind_group_read_mac_grid,
            bind_group_renderer,

            bind_group_density_projection_gather_error,

            pipeline_transfer_clear: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(layout_transfer_velocity.clone(), Path::new("simulation/transfer_clear.comp")),
            ),
            pipeline_transfer_build_linkedlist: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(layout_transfer_velocity.clone(), Path::new("simulation/transfer_build_linkedlist.comp")),
            ),
            pipeline_transfer_gather_velocity: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(layout_transfer_velocity.clone(), Path::new("simulation/transfer_gather_velocity.comp")),
            ),
            pipeline_divergence_compute: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(layout_divergence_compute.clone(), Path::new("simulation/divergence_compute.comp")),
            ),
            pipeline_divergence_remove: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(layout_write_velocity_volume.clone(), Path::new("simulation/divergence_remove.comp")),
            ),
            pipeline_extrapolate_velocity: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(layout_write_velocity_volume.clone(), Path::new("simulation/extrapolate_velocity.comp")),
            ),
            pipeline_advect_particles: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(layout_particles.clone(), Path::new("simulation/advect_particles.comp")),
            ),

            pipeline_density_projection_gather_error: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(
                    layout_density_projection_gather_error.clone(),
                    Path::new("simulation/density_projection_gather_error.comp"),
                ),
            ),

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

        // Clear velocities:
        // wgpu-rs doesn't zero initialize yet (bug/missing feature impl)
        // https://github.com/gfx-rs/wgpu/issues/563
        let offset_velocity_buffer = self.simulation_properties.num_particles as u64 * std::mem::size_of::<cgmath::Vector4<f32>>() as u64;
        let zero_velocity = vec![0 as u8; num_new_particles as usize * std::mem::size_of::<cgmath::Vector4<f32>>()];
        queue.write_buffer(&self.particles_velocity_x, offset_velocity_buffer, &zero_velocity);
        queue.write_buffer(&self.particles_velocity_y, offset_velocity_buffer, &zero_velocity);
        queue.write_buffer(&self.particles_velocity_z, offset_velocity_buffer, &zero_velocity);

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
                    .next_binding_vertex(binding_glsl::texture3D()) // marker
                    .next_binding_vertex(binding_glsl::texture3D()) // pressure
                    .next_binding_vertex(binding_glsl::texture3D()) // density
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

    const COMPUTE_LOCAL_SIZE_FLUID: wgpu::Extent3d = wgpu::Extent3d {
        width: 8,
        height: 8,
        depth: 8,
    };
    const COMPUTE_LOCAL_SIZE_PARTICLES: u32 = 512;

    pub fn step(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline_manager: &PipelineManager,
        queue: &wgpu::Queue,
        per_frame_bind_group: &wgpu::BindGroup,
    ) {
        if self.simulation_properties_dirty.get() {
            self.simulation_properties_uniformbuffer.update_content(queue, self.simulation_properties);
            self.simulation_properties_dirty.set(false);
        }

        let grid_work_groups = wgpu_utils::compute_group_size(self.grid_dimension, Self::COMPUTE_LOCAL_SIZE_FLUID);
        let particle_work_groups = wgpu_utils::compute_group_size_1d(self.simulation_properties.num_particles, Self::COMPUTE_LOCAL_SIZE_PARTICLES);

        let mut cpass = encoder.begin_compute_pass();
        cpass.set_bind_group(0, &per_frame_bind_group, &[]);
        cpass.set_bind_group(1, &self.bind_group_uniform, &[]);

        // mostly grouped by layouts.
        {
            for i in 0..3 {
                cpass.set_bind_group(2, &self.bind_group_transfer_velocity[i], &[]);

                // clear linkedlist grid & marker (only on first run)
                cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_transfer_clear));
                cpass.set_push_constants(0, &[i as u32]);
                cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);

                // Create particle linked lists by writing heads in dual grids
                cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_transfer_build_linkedlist));
                cpass.dispatch(particle_work_groups, 1, 1);

                // Gather velocities in velocity grid and apply global forces.
                cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_transfer_gather_velocity));
                cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);
            }
        }
        // Compute divergence & solve for pressure
        cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_divergence_compute));
        cpass.set_bind_group(1, &self.bind_group_divergence_compute, &[]); // Writes directly into Residual of the pressure solver.
        cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);
        self.pressure_solver.solve(&mut cpass, pipeline_manager);

        cpass.set_bind_group(1, &self.bind_group_uniform, &[]);
        {
            cpass.set_bind_group(2, &self.bind_group_write_velocity, &[]);

            // Make velocity grid divergence free
            cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_divergence_remove));
            cpass.set_bind_group(2, &self.bind_group_write_velocity, &[]);
            cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);

            // Extrapolate velocity
            // can only do a single extrapolation since we can't change cell types without double buffering
            // (this makes the extrapolation a bit heavier since it needs to sample all 8 diagonals as well)
            cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_extrapolate_velocity));
            cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);
        }
        {
            // Clear Marker & linked list.
            cpass.set_bind_group(2, &self.bind_group_transfer_velocity[0], &[]);
            cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_transfer_clear));
            cpass.set_push_constants(0, &[0]);
            cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);
        }
        {
            // Advect particles with grid and write new linked list for density gather.
            cpass.set_bind_group(2, &self.bind_group_advect_particles, &[]);
            cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_advect_particles));
            cpass.dispatch(particle_work_groups, 1, 1);
        }
        {
            // Compute density grid by another gather pass
            cpass.set_bind_group(2, &self.bind_group_density_projection_gather_error, &[]);
            cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_density_projection_gather_error));
            cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);
        }
    }
}
