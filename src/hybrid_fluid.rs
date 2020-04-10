use crate::wgpu_utils::binding_builder::*;
use crate::wgpu_utils::pipelines::*;
use crate::wgpu_utils::shader::*;
use crate::wgpu_utils::*;
use rand::prelude::*;
use std::{path::Path, rc::Rc};

pub struct HybridFluid {
    //gravity: cgmath::Vector3<f32>, // global gravity force in m/s² (== N/kg)
    grid_dimension: wgpu::Extent3d,

    particles: wgpu::Buffer,

    bind_group_write_particles: wgpu::BindGroup,
    bind_group_read_particles: wgpu::BindGroup,
    bind_group_velocity_grids: [wgpu::BindGroup; 2],

    pipeline_clear_grid: ReloadableComputePipeline,
    pipeline_velocity_to_grid: ReloadableComputePipeline,
    pipeline_velocity_to_particles: ReloadableComputePipeline,

    num_particles: u64,
    max_num_particles: u64,
}

// todo: probably want to split this up into several buffers
#[repr(C)]
#[derive(Clone, Copy)]
struct Particle {
    // Particle positions are in grid space to simplify shader computation
    // (no scaling/translation needed until we're rendering or interacting with other objects!)
    position: cgmath::Point3<f32>,
    padding: f32,
}

impl HybridFluid {
    // particles are distributed 2x2x2 within a single gridcell
    // (seems to be widely accepted as the default)
    const PARTICLES_PER_GRID_CELL: u32 = 8;

    // TODO: Split up/simplify all this binding generation code!
    pub fn new(
        device: &wgpu::Device,
        grid_dimension: wgpu::Extent3d,
        max_num_particles: u64,
        shader_dir: &ShaderDirectory,
        per_frame_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let group_layout_read_particles = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::buffer(true))
            .create(device, "BindGroupLayout: ParticlesReadOnly");
        let group_layout_write_particles = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::buffer(false))
            .create(device, "BindGroupLayout: ParticlesReadWrite");
        let group_layout_volumes = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::image3d(wgpu::TextureFormat::Rgba32Float, false))
            .next_binding_compute(binding_glsl::texture3D())
            .create(device, "BindGroupLayout: VelocityGrids");

        let velocity_texture_desc = wgpu::TextureDescriptor {
            label: Some("Texture: Velocity Grid"),
            size: grid_dimension,
            array_layer_count: 1,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::STORAGE,
        };
        let velocity_grids = [
            device.create_texture(&velocity_texture_desc),
            device.create_texture(&velocity_texture_desc),
        ];
        let texture_view_velocity_grids = [
            velocity_grids[0].create_view(&default_textureview(&velocity_texture_desc)),
            velocity_grids[1].create_view(&default_textureview(&velocity_texture_desc)),
        ];

        let bind_group_velocity_grids = [
            BindGroupBuilder::new(&group_layout_volumes)
                .texture(&texture_view_velocity_grids[0])
                .texture(&texture_view_velocity_grids[1])
                .create(device, "BindGroup: VelocityGrids2"),
            BindGroupBuilder::new(&group_layout_volumes)
                .texture(&texture_view_velocity_grids[1])
                .texture(&texture_view_velocity_grids[0])
                .create(device, "BindGroup: VelocityGrids1"),
        ];

        let particle_buffer_size = max_num_particles * std::mem::size_of::<Particle>() as u64;
        let particles = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Buffer: ParticleBuffer"),
            size: particle_buffer_size,
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::STORAGE_READ | wgpu::BufferUsage::COPY_DST,
        });

        let bind_group_write_particles = BindGroupBuilder::new(&group_layout_write_particles)
            .buffer(&particles, 0..particle_buffer_size)
            .create(device, "BindGroup: ParticlesReadWrite");
        let bind_group_read_particles = BindGroupBuilder::new(&group_layout_read_particles)
            .buffer(&particles, 0..particle_buffer_size)
            .create(device, "BindGroup: ParticlesReadOnly");

        let layout_write_particles = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[
                per_frame_bind_group_layout,
                &group_layout_write_particles.layout,
                &group_layout_volumes.layout,
            ],
        }));
        let layout_read_particles = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[
                per_frame_bind_group_layout,
                &group_layout_read_particles.layout,
                &group_layout_volumes.layout,
            ],
        }));

        HybridFluid {
            //gravity: cgmath::Vector3::new(0.0, -9.81, 0.0), // there needs to be some grid->world relation
            grid_dimension,

            particles,
            bind_group_write_particles,
            bind_group_read_particles,
            bind_group_velocity_grids,

            pipeline_clear_grid: ReloadableComputePipeline::new(device, &layout_write_particles, shader_dir, Path::new("clear_grid.comp")),
            pipeline_velocity_to_grid: ReloadableComputePipeline::new(device, &layout_read_particles, shader_dir, Path::new("velocity_to_grid.comp")),
            pipeline_velocity_to_particles: ReloadableComputePipeline::new(
                device,
                &layout_write_particles,
                shader_dir,
                Path::new("velocity_to_particles.comp"),
            ),

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

    pub fn try_reload_shaders(&mut self, device: &wgpu::Device, shader_dir: &ShaderDirectory) {
        let _ = self.pipeline_clear_grid.try_reload_shader(device, shader_dir);
        let _ = self.pipeline_velocity_to_grid.try_reload_shader(device, shader_dir);
        let _ = self.pipeline_velocity_to_particles.try_reload_shader(device, shader_dir);
    }

    // Adds a cube of fluid. Coordinates are in grid space! Very slow operation!
    // todo: Removes all previously added particles.
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

        let num_new_particles = self
            .max_num_particles
            .min(((max_grid.x - min_grid.x) * (max_grid.y - min_grid.y) * (max_grid.z - min_grid.z) * Self::PARTICLES_PER_GRID_CELL) as u64);

        let particle_size = std::mem::size_of::<Particle>() as u64;
        let particle_buffer_mapping = device.create_buffer_mapped(&wgpu::BufferDescriptor {
            label: Some("Buffer: Particle Update"),
            size: num_new_particles * particle_size,
            usage: wgpu::BufferUsage::COPY_SRC,
        });

        // Fill buffer with particle data
        let mut rng: rand::rngs::SmallRng = rand::SeedableRng::seed_from_u64(num_new_particles as u64);
        let new_particles =
            unsafe { std::slice::from_raw_parts_mut(particle_buffer_mapping.data.as_mut_ptr() as *mut Particle, num_new_particles as usize) };
        for (i, position) in new_particles.iter_mut().enumerate() {
            //let sample_idx = i as u32 % Self::PARTICLES_PER_GRID_CELL;
            let cell = cgmath::Point3::new(
                (i as u32 / Self::PARTICLES_PER_GRID_CELL % extent_cell.x) as f32,
                (i as u32 / Self::PARTICLES_PER_GRID_CELL / extent_cell.x % extent_cell.y) as f32,
                (i as u32 / Self::PARTICLES_PER_GRID_CELL / extent_cell.x / extent_cell.y) as f32,
            );
            *position = Particle {
                position: (cell + rng.gen::<cgmath::Vector3<f32>>()),
                padding: i as f32,
            };
        }

        init_encoder.copy_buffer_to_buffer(
            &particle_buffer_mapping.finish(),
            0,
            &self.particles,
            self.num_particles * particle_size,
            num_new_particles * particle_size,
        );
        self.num_particles += num_new_particles;
    }

    pub fn num_particles(&self) -> u64 {
        self.num_particles
    }

    pub fn particle_binding_resource(&self) -> wgpu::BindingResource {
        wgpu::BindingResource::Buffer {
            buffer: &self.particles,
            range: 0..self.particle_buffer_size(),
        }
    }

    pub fn particle_buffer_size(&self) -> u64 {
        self.max_num_particles as u64 * std::mem::size_of::<Particle>() as u64
    }

    // todo: timing
    pub fn step<'a>(&'a self, cpass: &mut wgpu::ComputePass<'a>) {
        // note on setting bind groups:
        // As of writing, webgpu-rs silently does nothing on dispatch if pipeline layout doesn't match currently set bind groups.
        // Leaving out set_bind_group for already bound resources is fine, but this will then also not emit any barrier at all!
        // TODO: Create a ticket on this? Is this a bug that needs reporting?

        // clear grid
        // It's either this or a loop over encoder.begin_render_pass which then also requires a myriad of texture views...
        // (might still be faster because RT clear operations are usually very quick :/)
        cpass.set_pipeline(self.pipeline_clear_grid.pipeline());
        cpass.set_bind_group(1, &self.bind_group_write_particles, &[]);
        cpass.set_bind_group(2, &self.bind_group_velocity_grids[0], &[]);
        cpass.dispatch(self.grid_dimension.width, self.grid_dimension.height, self.grid_dimension.depth);

        // Transfer velocities to grid. (write grid, read particles)
        cpass.set_pipeline(self.pipeline_velocity_to_grid.pipeline());
        cpass.set_bind_group(1, &self.bind_group_read_particles, &[]);
        cpass.set_bind_group(2, &self.bind_group_velocity_grids[0], &[]);
        cpass.dispatch(self.num_particles as u32, 1, 1);

        // Apply global forces (write grid)

        // Resolves forces on grid. (write grid, read grid)

        // Transfer velocities to particles. (read grid, write particles)
        cpass.set_pipeline(self.pipeline_velocity_to_particles.pipeline());
        cpass.set_bind_group(1, &self.bind_group_write_particles, &[]);
        cpass.set_bind_group(2, &self.bind_group_velocity_grids[1], &[]);
        cpass.dispatch(self.num_particles as u32, 1, 1);

        // Advect particles.  (write particles)
    }
}
