use crate::hybrid_fluid::HybridFluid;
use crate::wgpu_utils::{pipelines::PipelineManager, shader::ShaderDirectory};

// Scene data & simulation.
pub struct Scene {
    hybrid_fluid: HybridFluid,
    pub fluid_origin: cgmath::Point3<f32>,
    pub fluid_to_world_scale: f32,
}

impl Scene {
    pub fn new(
        device: &wgpu::Device,
        init_encoder: &mut wgpu::CommandEncoder,
        shader_dir: &ShaderDirectory,
        pipeline_manager: &mut PipelineManager,
        per_frame_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let grid_dimension = wgpu::Extent3d {
            width: 128,
            height: 64,
            depth: 64,
        };

        let mut hybrid_fluid = HybridFluid::new(device, grid_dimension, 2000000, shader_dir, pipeline_manager, per_frame_bind_group_layout);

        hybrid_fluid.add_fluid_cube(
            device,
            init_encoder,
            cgmath::Point3::new(1.0, 1.0, 1.0),
            cgmath::Point3::new(64.0, 40.0, 64.0),
        );

        Scene {
            hybrid_fluid,
            fluid_origin: cgmath::point3(0.0, 0.0, 0.0),
            fluid_to_world_scale: 1.0,
        }
    }

    pub fn step<'a>(&'a self, cpass: &mut wgpu::ComputePass<'a>, pipeline_manager: &'a PipelineManager) {
        self.hybrid_fluid.step(cpass, pipeline_manager);
    }

    pub fn fluid(&self) -> &HybridFluid {
        &self.hybrid_fluid
    }
}
