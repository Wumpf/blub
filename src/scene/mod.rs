pub mod models;
pub mod voxelization;

use crate::{
    simulation::HybridFluid,
    timer::Timer,
    wgpu_utils::{pipelines::PipelineManager, shader::ShaderDirectory},
};
use wgpu_profiler::{wgpu_profiler, GpuProfiler};

use serde::Deserialize;
use std::{error, fs::File, io::BufReader, path::Path, path::PathBuf};

use self::{
    models::{SceneModels, StaticObjectConfig},
    voxelization::SceneVoxelization,
};

#[derive(Deserialize)]
pub struct Box {
    pub min: cgmath::Point3<f32>,
    pub max: cgmath::Point3<f32>,
}

// Data describing a fluid in the scene.
#[derive(Deserialize)]
pub struct FluidConfig {
    pub world_position: cgmath::Point3<f32>,
    pub grid_to_world_scale: f32,
    pub grid_dimension: cgmath::Point3<u32>,
    pub max_num_particles: u32,
    pub fluid_cubes: Vec<Box>,
}

// Data describing a scene.
#[derive(Deserialize)]
pub struct SceneConfig {
    // global gravity (in world space)
    pub gravity: cgmath::Vector3<f32>,
    pub fluid: FluidConfig,
    #[serde(default)]
    pub static_objects: Vec<StaticObjectConfig>,
}

// Scene data & simulation.
pub struct Scene {
    hybrid_fluid: HybridFluid,
    config: SceneConfig,
    pub models: SceneModels,
    pub voxelization: SceneVoxelization,
    distance_field_dirty: bool,
    path: PathBuf,
}

impl Scene {
    pub fn new(
        path: &Path,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        shader_dir: &ShaderDirectory,
        pipeline_manager: &mut PipelineManager,
        global_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Result<Self, std::boxed::Box<dyn error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let config: SceneConfig = serde_json::from_reader(reader)?;

        let voxelization = SceneVoxelization::new(
            device,
            shader_dir,
            pipeline_manager,
            global_bind_group_layout,
            wgpu::Extent3d {
                width: config.fluid.grid_dimension.x,
                height: config.fluid.grid_dimension.y,
                depth_or_array_layers: config.fluid.grid_dimension.z,
            },
        );

        let hybrid_fluid = Self::create_fluid_from_config(
            &config,
            device,
            queue,
            shader_dir,
            pipeline_manager,
            global_bind_group_layout,
            &voxelization,
        );
        let models = SceneModels::from_config(&device, queue, &config.static_objects, &config.fluid)?;

        Ok(Scene {
            hybrid_fluid,
            config,
            models,
            voxelization,
            distance_field_dirty: true,
            path: path.to_path_buf(),
        })
    }

    pub fn config(&self) -> &SceneConfig {
        &self.config
    }

    pub fn num_active_particles(&self) -> u32 {
        self.hybrid_fluid.num_active_particles()
    }

    fn create_fluid_from_config(
        config: &SceneConfig,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        shader_dir: &ShaderDirectory,
        pipeline_manager: &mut PipelineManager,
        global_bind_group_layout: &wgpu::BindGroupLayout,
        voxelization: &SceneVoxelization,
    ) -> HybridFluid {
        let mut hybrid_fluid = HybridFluid::new(
            device,
            wgpu::Extent3d {
                width: config.fluid.grid_dimension.x,
                height: config.fluid.grid_dimension.y,
                depth_or_array_layers: config.fluid.grid_dimension.z,
            },
            config.fluid.max_num_particles,
            shader_dir,
            pipeline_manager,
            global_bind_group_layout,
            voxelization,
        );

        for cube in config.fluid.fluid_cubes.iter() {
            hybrid_fluid.add_fluid_cube(
                queue,
                cube.min / config.fluid.grid_to_world_scale,
                cube.max / config.fluid.grid_to_world_scale,
            );
        }
        hybrid_fluid.set_gravity_grid(config.gravity / config.fluid.grid_to_world_scale);

        // Creating the fluid is quite heavy, make sure we're done with all the buffer book-keeping before we move on.
        device.poll(wgpu::Maintain::Wait);
        hybrid_fluid
    }

    pub fn reset(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        shader_dir: &ShaderDirectory,
        pipeline_manager: &mut PipelineManager,
        global_bind_group_layout: &wgpu::BindGroupLayout,
    ) {
        self.hybrid_fluid = Self::create_fluid_from_config(
            &self.config,
            device,
            queue,
            shader_dir,
            pipeline_manager,
            global_bind_group_layout,
            &self.voxelization,
        );
        self.distance_field_dirty = true;
    }

    pub fn step(
        &mut self,
        timer: &Timer,
        device: &wgpu::Device,
        profiler: &mut GpuProfiler,
        pipeline_manager: &PipelineManager,
        queue: &wgpu::Queue,
        global_bind_group: &wgpu::BindGroup,
    ) {
        if self.distance_field_dirty {
            self.hybrid_fluid.update_signed_distance_field_for_static(
                device,
                pipeline_manager,
                queue,
                global_bind_group,
                &self.models.meshes,
                &self.path,
            );
            self.distance_field_dirty = false;
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Encoder: Scene Step"),
        });

        //wgpu_profiler!("Animate Models", profiler, &mut encoder, device, {
        self.models.step(timer, queue, &self.config.fluid);
        //});

        wgpu_profiler!("Voxelize Scene", profiler, &mut encoder, device, {
            self.voxelization.update(&mut encoder, pipeline_manager, global_bind_group, &self.models);
        });

        wgpu_profiler!("HybridFluid step", profiler, &mut encoder, device, {
            self.hybrid_fluid.step(
                timer.simulation_delta(),
                &mut encoder,
                device,
                queue,
                global_bind_group,
                pipeline_manager,
                profiler,
            );
        });
        profiler.resolve_queries(&mut encoder);
        queue.submit(Some(encoder.finish()));
        profiler.end_frame().unwrap();
        self.hybrid_fluid.update_statistics();
    }

    pub fn fluid(&self) -> &HybridFluid {
        &self.hybrid_fluid
    }

    pub fn fluid_mut(&mut self) -> &mut HybridFluid {
        &mut self.hybrid_fluid
    }
}
