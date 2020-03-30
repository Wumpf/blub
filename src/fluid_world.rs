use rand::prelude::*;

pub struct FluidWorld {
    //gravity: cgmath::Vector3<f32>, // global gravity force in m/s² (== N/kg)
    grid_dimension: cgmath::Vector3<u32>,
    grid_cellsize: f32,

    particles: wgpu::Buffer,
    num_particles: u32,
}

// todo: probably want to split this up into several buffers
#[repr(C)]
#[derive(Clone, Copy)]
struct Particle {
    position: cgmath::Point3<f32>,
    padding: f32,
}

impl FluidWorld {
    // particles are distributed 2x2x2 within a single gridcell
    // (seems to be widely accepted as the default)
    const PARTICLES_PER_GRID_CELL: u32 = 8;

    pub fn new(device: &wgpu::Device, grid_dimension: cgmath::Vector3<u32>, grid_cellsize: f32) -> Self {
        FluidWorld {
            //gravity: cgmath::Vector3::new(0.0, -9.81, 0.0),
            grid_dimension,
            grid_cellsize,

            // dummy. is there a invalid buffer?
            particles: device.create_buffer(&wgpu::BufferDescriptor {
                size: 1,
                usage: wgpu::BufferUsage::STORAGE,
            }),
            num_particles: 0,
        }
    }

    fn clamp_to_grid(&self, grid_cor: cgmath::Point3<f32>) -> cgmath::Point3<u32> {
        cgmath::Point3::new(
            self.grid_dimension.x.min(grid_cor.x as u32),
            self.grid_dimension.y.min(grid_cor.y as u32),
            self.grid_dimension.z.min(grid_cor.z as u32),
        )
    }

    // Adds a cube of fluid. Very slow operation!
    // todo: Removes all previously added particles.
    pub fn add_fluid_cube(&mut self, device: &wgpu::Device, min: cgmath::Point3<f32>, max: cgmath::Point3<f32>) {
        // align to whole cells for simplicity.
        let min_cell = self.clamp_to_grid(min / self.grid_cellsize + cgmath::vec3(0.5, 0.5, 0.5));
        let min_max = min_cell + cgmath::vec3(1, 1, 1);
        let max_tmp = self.clamp_to_grid(max / self.grid_cellsize + cgmath::vec3(0.5, 0.5, 0.5));
        let max_cell = cgmath::Point3::new(max_tmp.x.max(min_max.x), max_tmp.y.max(min_max.y), max_tmp.z.max(min_max.z));
        let extent_cell = max_cell - min_cell;

        let num_new_particles = (max_cell.x - min_cell.x) * (max_cell.y - min_cell.y) * (max_cell.z - min_cell.z) * Self::PARTICLES_PER_GRID_CELL;

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
                position: (cell + rng.gen::<cgmath::Vector3<f32>>()) * self.grid_cellsize,
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
    pub fn step(&self) {}
}
