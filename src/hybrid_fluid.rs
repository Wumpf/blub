use crate::shader::*;
use crate::wgpu_utils::*;
use rand::prelude::*;
use std::path::Path;

struct HybridFluidComputePipelines {
    transfer_velocity_to_grid: wgpu::ComputePipeline,
    transfer_velocity_to_particles: wgpu::ComputePipeline,
}

impl HybridFluidComputePipelines {
    fn new(
        device: &wgpu::Device,
        pipeline_layout_write_particles: &wgpu::PipelineLayout,
        pipeline_layout_read_particles: &wgpu::PipelineLayout,
        shader_dir: &ShaderDirectory,
    ) -> Option<Self> {
        let shader_transfer_velocity_to_grid = shader_dir.load_shader_module(device, Path::new("transfer_velocity_to_grid.comp"))?;
        let shader_transfer_velocity_to_particles = shader_dir.load_shader_module(device, Path::new("transfer_velocity_to_particles.comp"))?;

        Some(HybridFluidComputePipelines {
            transfer_velocity_to_grid: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                layout: pipeline_layout_write_particles,
                compute_stage: wgpu::ProgrammableStageDescriptor {
                    module: &shader_transfer_velocity_to_grid,
                    entry_point: SHADER_ENTRY_POINT_NAME,
                },
            }),
            transfer_velocity_to_particles: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                layout: pipeline_layout_read_particles,
                compute_stage: wgpu::ProgrammableStageDescriptor {
                    module: &shader_transfer_velocity_to_particles,
                    entry_point: SHADER_ENTRY_POINT_NAME,
                },
            }),
        })
    }
}

pub struct HybridFluid {
    //gravity: cgmath::Vector3<f32>, // global gravity force in m/s² (== N/kg)
    grid_dimension: wgpu::Extent3d,

    particles: wgpu::Buffer,
    bind_group_write_particles: wgpu::BindGroup,
    bind_group_read_particles: wgpu::BindGroup,

    velocity_grids: [wgpu::Texture; 2],
    bind_group_velocity_grids: [wgpu::BindGroup; 2],

    pipeline_layout_write_particles: wgpu::PipelineLayout,
    pipeline_layout_read_particles: wgpu::PipelineLayout,

    compute_pipelines: HybridFluidComputePipelines,

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

// #[repr(C)]
// #[derive(Clone, Copy)]
// pub struct HybridFluidUniformBufferContent {
//     pub num_particles: u32,
// }
// pub type HybridFluidUniformBuffer = UniformBuffer<HybridFluidUniformBufferContent>;

impl HybridFluid {
    // particles are distributed 2x2x2 within a single gridcell
    // (seems to be widely accepted as the default)
    const PARTICLES_PER_GRID_CELL: u32 = 8;

    // TODO: Split up/simplify all this binding generation code!
    pub fn new(device: &wgpu::Device, grid_dimension: wgpu::Extent3d, max_num_particles: u64, shader_dir: &ShaderDirectory) -> Self {
        let group_layout_read_particles = BindGroupLayoutBuilder::new()
            .next_binding_compute(bindingtype_storagebuffer_readonly())
            .create(device);
        let group_layout_write_particles = BindGroupLayoutBuilder::new()
            .next_binding_compute(bindingtype_storagebuffer_readwrite())
            .create(device);
        let group_layout_volumes = BindGroupLayoutBuilder::new()
            .next_binding_compute(bindingtype_storagetexture_3d())
            .next_binding_compute(bindingtype_texture_3d())
            .create(device);

        let pipeline_layout_write_particles = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&group_layout_write_particles.layout, &group_layout_volumes.layout],
        });
        let pipeline_layout_read_particles = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&group_layout_read_particles.layout, &group_layout_volumes.layout],
        });
        let compute_pipelines =
            HybridFluidComputePipelines::new(device, &pipeline_layout_write_particles, &pipeline_layout_read_particles, shader_dir).unwrap();

        let velocity_texture_desc = wgpu::TextureDescriptor {
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
                .resource(wgpu::BindingResource::TextureView(&texture_view_velocity_grids[0]))
                .resource(wgpu::BindingResource::TextureView(&texture_view_velocity_grids[1]))
                .create(device),
            BindGroupBuilder::new(&group_layout_volumes)
                .resource(wgpu::BindingResource::TextureView(&texture_view_velocity_grids[1]))
                .resource(wgpu::BindingResource::TextureView(&texture_view_velocity_grids[0]))
                .create(device),
        ];

        let particle_buffer_size = max_num_particles * std::mem::size_of::<Particle>() as u64;
        let particles = device.create_buffer(&wgpu::BufferDescriptor {
            size: particle_buffer_size,
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::STORAGE_READ | wgpu::BufferUsage::COPY_DST,
        });

        let particle_resourceview = wgpu::BindingResource::Buffer {
            buffer: &particles,
            range: 0..particle_buffer_size,
        };
        let bind_group_write_particles = BindGroupBuilder::new(&group_layout_write_particles)
            .resource(particle_resourceview.clone())
            .create(device);
        let bind_group_read_particles = BindGroupBuilder::new(&group_layout_read_particles)
            .resource(particle_resourceview.clone())
            .create(device);

        HybridFluid {
            //gravity: cgmath::Vector3::new(0.0, -9.81, 0.0), // there needs to be some grid->world relation
            grid_dimension,

            particles,
            bind_group_write_particles,
            bind_group_read_particles,

            velocity_grids,
            bind_group_velocity_grids,

            pipeline_layout_write_particles,
            pipeline_layout_read_particles,
            compute_pipelines,

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
        if let Some(pipelines) = HybridFluidComputePipelines::new(
            device,
            &self.pipeline_layout_write_particles,
            &self.pipeline_layout_read_particles,
            shader_dir,
        ) {
            self.compute_pipelines = pipelines;
        }
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

        let particle_buffer_mapping = device.create_buffer_mapped(num_new_particles as usize, wgpu::BufferUsage::COPY_SRC);

        // Fill buffer with particle data
        let mut rng: rand::rngs::SmallRng = rand::SeedableRng::seed_from_u64(num_new_particles as u64);
        for (i, position) in particle_buffer_mapping.data.iter_mut().enumerate() {
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

        let particle_size = std::mem::size_of::<Particle>() as u64;
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
    pub fn step(&self, cpass: &mut wgpu::ComputePass) {
        // Transfer velocities to grid. (write grid, read particles)
        cpass.set_bind_group(0, &self.bind_group_read_particles, &[]);
        cpass.set_bind_group(1, &self.bind_group_velocity_grids[0], &[]);
        cpass.set_pipeline(&self.compute_pipelines.transfer_velocity_to_grid);
        // cpass.dispatch(self.num_particles, 1, 1);

        // Apply global forces (write grid)

        // Resolves forces on grid. (write grid, read grid)

        // Transfer velocities to particles. (read grid, write particles)
        cpass.set_bind_group(0, &self.bind_group_write_particles, &[]);
        cpass.set_bind_group(1, &self.bind_group_velocity_grids[0], &[]);
        cpass.set_pipeline(&self.compute_pipelines.transfer_velocity_to_particles);
        // cpass.dispatch(self.num_particles, 1, 1);

        // Advect particles.  (write particles)
    }
}
