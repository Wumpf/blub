use super::shader::*;
use rand::prelude::*;
use std::path::Path;

struct HybridFluidComputePipelines {
    transfer_velocity_to_grid: wgpu::ComputePipeline,
    transfer_velocity_to_particles: wgpu::ComputePipeline,
}

impl HybridFluidComputePipelines {
    fn new(device: &wgpu::Device, pipeline_layout: &wgpu::PipelineLayout, shader_dir: &ShaderDirectory) -> Option<Self> {
        let shader_transfer_velocity_to_grid = shader_dir.load_shader_module(device, Path::new("transfer_velocity_to_grid.comp"))?;
        let shader_transfer_velocity_to_particles = shader_dir.load_shader_module(device, Path::new("transfer_velocity_to_particles.comp"))?;

        Some(HybridFluidComputePipelines {
            transfer_velocity_to_grid: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                layout: pipeline_layout,
                compute_stage: wgpu::ProgrammableStageDescriptor {
                    module: &shader_transfer_velocity_to_grid,
                    entry_point: SHADER_ENTRY_POINT_NAME,
                },
            }),
            transfer_velocity_to_particles: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                layout: pipeline_layout,
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
    mac_grid_vx: wgpu::Texture,
    mac_grid_vy: wgpu::Texture,
    mac_grid_vz: wgpu::Texture,

    pipeline_layout: wgpu::PipelineLayout,
    compute_pipelines: HybridFluidComputePipelines,

    num_particles: u32,
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

    pub fn new(device: &wgpu::Device, grid_dimension: wgpu::Extent3d, shader_dir: &ShaderDirectory) -> Self {
        let grid_component_desc = wgpu::TextureDescriptor {
            size: grid_dimension,
            array_layer_count: 1,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::STORAGE,
        };

        // todo: Structure this stuff in a more clever way.
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            bindings: &[
                // Particle buffer
                wgpu::BindGroupLayoutBinding {
                    binding: 1,
                    visibility: wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::StorageBuffer {
                        dynamic: false,
                        readonly: true,
                    },
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&bind_group_layout],
        });
        let compute_pipelines = HybridFluidComputePipelines::new(device, &pipeline_layout, shader_dir).unwrap();

        HybridFluid {
            //gravity: cgmath::Vector3::new(0.0, -9.81, 0.0), // there needs to be some grid->world relation
            grid_dimension,

            // dummy. is there an invalid buffer type?
            particles: device.create_buffer(&wgpu::BufferDescriptor {
                size: 1,
                usage: wgpu::BufferUsage::STORAGE,
            }),
            mac_grid_vx: device.create_texture(&grid_component_desc),
            mac_grid_vy: device.create_texture(&grid_component_desc),
            mac_grid_vz: device.create_texture(&grid_component_desc),

            pipeline_layout,
            compute_pipelines,

            num_particles: 0,
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
        if let Some(pipelines) = HybridFluidComputePipelines::new(device, &self.pipeline_layout, shader_dir) {
            self.compute_pipelines = pipelines;
        }
    }

    // Adds a cube of fluid. Coordinates are in grid space! Very slow operation!
    // todo: Removes all previously added particles.
    pub fn add_fluid_cube(&mut self, device: &wgpu::Device, min_grid: cgmath::Point3<f32>, max_grid: cgmath::Point3<f32>) {
        // align to whole cells for simplicity.
        let min_grid = self.clamp_to_grid(min_grid);
        let max_grid = self.clamp_to_grid(max_grid);
        let extent_cell = max_grid - min_grid;

        let num_new_particles = (max_grid.x - min_grid.x) * (max_grid.y - min_grid.y) * (max_grid.z - min_grid.z) * Self::PARTICLES_PER_GRID_CELL;

        // TODO: Keep previous particles! Maybe just have a max particle num on creation and keep track of how many we actually use.
        self.num_particles = num_new_particles;
        let particle_buffer_mapping =
            device.create_buffer_mapped(self.num_particles as usize, wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::STORAGE_READ);

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

        self.particles = particle_buffer_mapping.finish();
    }

    pub fn num_particles(&self) -> u32 {
        self.num_particles
    }

    pub fn particle_buffer(&self) -> &wgpu::Buffer {
        &self.particles
    }

    pub fn particle_buffer_size(&self) -> u64 {
        self.num_particles as u64 * std::mem::size_of::<Particle>() as u64
    }

    // todo: timing
    pub fn step(&self, cpass: &mut wgpu::ComputePass) {
        // Transfer velocities to grid.
        // cpass.set_pipeline(pipeline);
        // cpass.set_bind_group(index, bind_group, offsets);
        // cpass.dispatch(self.num_particles, 1, 1);

        // Resolves forces on grid.

        // Transfer velocities to particles.

        // Advect particles.
    }
}
