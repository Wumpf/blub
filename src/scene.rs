use crate::hybrid_fluid::HybridFluid;
use crate::wgpu_utils::{pipelines::PipelineManager, shader::ShaderDirectory};

use serde::Deserialize;
use std::{
    fs::File,
    io::{self, BufReader},
    path::Path,
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
}

// Scene data & simulation.
pub struct Scene {
    hybrid_fluid: HybridFluid,
    pub config: SceneConfig,
}

impl Scene {
    pub fn new(
        scene_path: &Path,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        shader_dir: &ShaderDirectory,
        pipeline_manager: &mut PipelineManager,
        per_frame_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Result<Self, io::Error> {
        let file = File::open(scene_path)?;
        let reader = BufReader::new(file);
        let config: SceneConfig = serde_json::from_reader(reader)?;

        let hybrid_fluid = Self::create_fluid_from_config(&config, device, queue, shader_dir, pipeline_manager, per_frame_bind_group_layout);

        Ok(Scene { hybrid_fluid, config })
    }

    fn create_fluid_from_config(
        config: &SceneConfig,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        shader_dir: &ShaderDirectory,
        pipeline_manager: &mut PipelineManager,
        per_frame_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> HybridFluid {
        let mut init_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("HybridFluid Init Encoder"),
        });

        let mut hybrid_fluid = HybridFluid::new(
            device,
            wgpu::Extent3d {
                width: config.fluid.grid_dimension.x,
                height: config.fluid.grid_dimension.y,
                depth: config.fluid.grid_dimension.z,
            },
            config.fluid.max_num_particles,
            shader_dir,
            pipeline_manager,
            per_frame_bind_group_layout,
        );

        for cube in config.fluid.fluid_cubes.iter() {
            hybrid_fluid.add_fluid_cube(
                device,
                &mut init_encoder,
                cube.min / config.fluid.grid_to_world_scale,
                cube.max / config.fluid.grid_to_world_scale,
            );
        }
        queue.submit(Some(init_encoder.finish()));
        hybrid_fluid.set_gravity_grid(config.gravity / config.fluid.grid_to_world_scale);
        hybrid_fluid
    }

    pub fn reset(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        shader_dir: &ShaderDirectory,
        pipeline_manager: &mut PipelineManager,
        per_frame_bind_group_layout: &wgpu::BindGroupLayout,
    ) {
        self.hybrid_fluid = Self::create_fluid_from_config(&self.config, device, queue, shader_dir, pipeline_manager, per_frame_bind_group_layout);
    }

    pub fn step<'a>(&'a self, cpass: &mut wgpu::ComputePass<'a>, pipeline_manager: &'a PipelineManager, queue: &wgpu::Queue) {
        self.hybrid_fluid.step(cpass, pipeline_manager, queue);
    }

    pub fn fluid(&self) -> &HybridFluid {
        &self.hybrid_fluid
    }
}
