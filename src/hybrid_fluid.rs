use crate::wgpu_utils::binding_builder::*;
use crate::wgpu_utils::pipelines::*;
use crate::wgpu_utils::shader::*;
use crate::wgpu_utils::*;
use rand::prelude::*;
use std::{path::Path, rc::Rc};
use uniformbuffer::PaddedVector3;

pub struct HybridFluid {
    //gravity: cgmath::Vector3<f32>, // global gravity force in m/s² (== N/kg)
    grid_dimension: wgpu::Extent3d,

    particles: wgpu::Buffer,

    bind_group_particles_rw: wgpu::BindGroup,
    bind_group_particles_ro: wgpu::BindGroup,
    bind_group_vgrids: [wgpu::BindGroup; 2],
    bind_group_llgrid_rw: wgpu::BindGroup,
    bind_group_llgrid_ro: wgpu::BindGroup,

    pipeline_clear_grids: ReloadableComputePipeline,
    pipeline_build_llgrid: ReloadableComputePipeline,
    pipeline_build_vgrid: ReloadableComputePipeline,
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
    linked_list_next: u32,
    velocity: PaddedVector3,
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
        let group_layout_particles_ro = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::buffer(true))
            .create(device, "BindGroupLayout: ParticlesReadOnly");
        let group_layout_particles_rw = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::buffer(false))
            .create(device, "BindGroupLayout: ParticlesReadWrite");
        let group_layout_volumes = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::image3d(wgpu::TextureFormat::Rgba32Float, false))
            .next_binding_compute(binding_glsl::texture3D())
            .create(device, "BindGroupLayout: VelocityGrids");
        let group_layout_llgrid_rw = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::uimage3d(wgpu::TextureFormat::R32Uint, false))
            .create(device, "BindGroupLayout: Linked List Dual Grid ReadWrite");
        let group_layout_llgrid_ro = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::uimage3d(wgpu::TextureFormat::R32Uint, true))
            .create(device, "BindGroupLayout: Linked List Dual Grid ReadOnly");

        let bind_group_vgrids = {
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
            let vgrids = [
                device.create_texture(&velocity_texture_desc),
                device.create_texture(&velocity_texture_desc),
            ];
            let texture_view_vgrids = [vgrids[0].create_default_view(), vgrids[1].create_default_view()];
            [
                BindGroupBuilder::new(&group_layout_volumes)
                    .texture(&texture_view_vgrids[0])
                    .texture(&texture_view_vgrids[1])
                    .create(device, "BindGroup: VelocityGrids2"),
                BindGroupBuilder::new(&group_layout_volumes)
                    .texture(&texture_view_vgrids[1])
                    .texture(&texture_view_vgrids[0])
                    .create(device, "BindGroup: VelocityGrids1"),
            ]
        };

        let llgrid = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Texture: Linked List Grid"),
            size: grid_dimension,
            array_layer_count: 1,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsage::STORAGE,
        });
        let llgrid_view = llgrid.create_default_view();
        let bind_group_llgrid_rw = BindGroupBuilder::new(&group_layout_llgrid_rw)
            .texture(&llgrid_view)
            .create(device, "BindGroup: LinkedListGridRW");
        let bind_group_llgrid_ro = BindGroupBuilder::new(&group_layout_llgrid_ro)
            .texture(&llgrid_view)
            .create(device, "BindGroup: LinkedListGridRO");

        let particle_buffer_size = max_num_particles * std::mem::size_of::<Particle>() as u64;
        let particles = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Buffer: ParticleBuffer"),
            size: particle_buffer_size,
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::STORAGE_READ | wgpu::BufferUsage::COPY_DST,
        });

        let bind_group_particles_rw = BindGroupBuilder::new(&group_layout_particles_rw)
            .buffer(&particles, 0..particle_buffer_size)
            .create(device, "BindGroup: ParticlesReadWrite");
        let bind_group_particles_ro = BindGroupBuilder::new(&group_layout_particles_ro)
            .buffer(&particles, 0..particle_buffer_size)
            .create(device, "BindGroup: ParticlesReadOnly");

        // Note that layouts directly correspond to DX12 root signatures.
        // We want to avoid having many of them and share as much as we can, but since WebGPU needs to set barriers for everything that is not readonly it's a tricky tradeoff.
        // Considering that all pipelines here require UAV barriers anyways a few more or less won't make too much difference (... is that true?).
        // Therefore we're compromising for less layouts & easier to maintain code.
        let layout_particles_to_grid = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[
                per_frame_bind_group_layout,
                &group_layout_particles_rw.layout, // not needed for clearing pass, ignored
                &group_layout_llgrid_rw.layout,
                &group_layout_volumes.layout,
            ],
        }));
        let layout_build_grids = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[
                per_frame_bind_group_layout,
                &group_layout_particles_ro.layout,
                &group_layout_llgrid_ro.layout, // only needed on first step, but keeping readonly bound shouldn't matter
                &group_layout_volumes.layout,
            ],
        }));
        let layout_grid_to_particles = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[
                per_frame_bind_group_layout,
                &group_layout_particles_rw.layout,
                &group_layout_llgrid_ro.layout, // not needed, but keeping readonly bound shouldn't matter
                &group_layout_volumes.layout,   // todo: Use pure read-only grid?
            ],
        }));

        HybridFluid {
            //gravity: cgmath::Vector3::new(0.0, -9.81, 0.0), // there needs to be some grid->world relation
            grid_dimension,

            particles,
            bind_group_particles_rw,
            bind_group_particles_ro,
            bind_group_vgrids,
            bind_group_llgrid_rw,
            bind_group_llgrid_ro,

            pipeline_clear_grids: ReloadableComputePipeline::new(device, &layout_particles_to_grid, shader_dir, Path::new("clear_grids.comp")),
            pipeline_build_llgrid: ReloadableComputePipeline::new(device, &layout_particles_to_grid, shader_dir, Path::new("build_llgrid.comp")),
            pipeline_build_vgrid: ReloadableComputePipeline::new(device, &layout_build_grids, shader_dir, Path::new("build_vgrid.comp")),
            pipeline_velocity_to_particles: ReloadableComputePipeline::new(
                device,
                &layout_grid_to_particles,
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
        let _ = self.pipeline_clear_grids.try_reload_shader(device, shader_dir);
        let _ = self.pipeline_build_llgrid.try_reload_shader(device, shader_dir);
        let _ = self.pipeline_build_vgrid.try_reload_shader(device, shader_dir);
        let _ = self.pipeline_velocity_to_particles.try_reload_shader(device, shader_dir);
    }

    // Clears all state (i.e. removes fluid particles)
    pub fn reset(&mut self) {
        self.num_particles = 0;
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
        for (i, particle) in new_particles.iter_mut().enumerate() {
            //let sample_idx = i as u32 % Self::PARTICLES_PER_GRID_CELL;
            let cell = cgmath::Point3::new(
                (i as u32 / Self::PARTICLES_PER_GRID_CELL % extent_cell.x) as f32,
                (i as u32 / Self::PARTICLES_PER_GRID_CELL / extent_cell.x % extent_cell.y) as f32,
                (i as u32 / Self::PARTICLES_PER_GRID_CELL / extent_cell.x / extent_cell.y) as f32,
            );
            let position = cell + rng.gen::<cgmath::Vector3<f32>>();
            *particle = Particle {
                position,
                linked_list_next: 0xFFFFFFFF,
                velocity: cgmath::vec3(0.0, 0.0, 0.0).into(),
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
        const COMPUTE_LOCAL_SIZE_FLUID: wgpu::Extent3d = wgpu::Extent3d {
            width: 8,
            height: 8,
            depth: 8,
        };
        const COMPUTE_LOCAL_SIZE_PARTICLES: u32 = 512;

        // grouped by layouts.
        {
            // clear pass doesn't need particle write access, but this reduces amount of needed layouts.
            cpass.set_bind_group(1, &self.bind_group_particles_rw, &[]);
            cpass.set_bind_group(2, &self.bind_group_llgrid_rw, &[]);
            cpass.set_bind_group(3, &self.bind_group_vgrids[0], &[]);

            // clear front velocity and linkedlist grid
            // It's either this or a loop over encoder.begin_render_pass which then also requires a myriad of texture views...
            // (might still be faster because RT clear operations are usually very quick :/)
            cpass.set_pipeline(self.pipeline_clear_grids.pipeline());
            cpass.dispatch(
                self.grid_dimension.width / COMPUTE_LOCAL_SIZE_FLUID.width,
                self.grid_dimension.height / COMPUTE_LOCAL_SIZE_FLUID.height,
                self.grid_dimension.depth / COMPUTE_LOCAL_SIZE_FLUID.depth,
            );

            // Create particle linked lists and write heads in dual grids
            // Transfer velocities to grid. (write grid, read particles)
            cpass.set_pipeline(self.pipeline_build_llgrid.pipeline());
            cpass.dispatch(self.num_particles as u32 / COMPUTE_LOCAL_SIZE_PARTICLES, 1, 1);
        }
        {
            cpass.set_bind_group(1, &self.bind_group_particles_ro, &[]);
            cpass.set_bind_group(2, &self.bind_group_llgrid_ro, &[]);

            // Gather velocities in velocity grid and apply global forces.
            cpass.set_pipeline(self.pipeline_build_vgrid.pipeline());
            cpass.dispatch(
                self.grid_dimension.width / COMPUTE_LOCAL_SIZE_FLUID.width,
                self.grid_dimension.height / COMPUTE_LOCAL_SIZE_FLUID.height,
                self.grid_dimension.depth / COMPUTE_LOCAL_SIZE_FLUID.depth,
            );

            // Resolves forces on grid. (write grid, read grid)
        }
        {
            // Transfer velocities to particles. (read grid, write particles)
            cpass.set_pipeline(self.pipeline_velocity_to_particles.pipeline());
            cpass.set_bind_group(1, &self.bind_group_particles_rw, &[]);
            cpass.set_bind_group(3, &self.bind_group_vgrids[1], &[]);
            cpass.dispatch(self.num_particles as u32 / COMPUTE_LOCAL_SIZE_PARTICLES, 1, 1);

            // Advect particles.  (write particles)
        }
    }
}
