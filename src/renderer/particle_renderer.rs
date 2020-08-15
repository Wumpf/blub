use crate::wgpu_utils::pipelines::*;
use crate::{
    render_output::{hdr_backbuffer::HdrBackbuffer, screen::Screen},
    simulation::HybridFluid,
    wgpu_utils::shader::*,
};
use std::{path::Path, rc::Rc};

pub struct ParticleRenderer {
    render_pipeline: RenderPipelineHandle,
}

impl ParticleRenderer {
    pub fn new(
        device: &wgpu::Device,
        shader_dir: &ShaderDirectory,
        pipeline_manager: &mut PipelineManager,
        per_frame_bind_group_layout: &wgpu::BindGroupLayout,
        fluid_renderer_group_layout: &wgpu::BindGroupLayout,
    ) -> ParticleRenderer {
        let render_pipeline = pipeline_manager.create_render_pipeline(
            device,
            shader_dir,
            RenderPipelineCreationDesc::new(
                Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("ParticleRenderer Pipeline Layout"),
                    bind_group_layouts: &[&per_frame_bind_group_layout, &fluid_renderer_group_layout],
                    push_constant_ranges: &[],
                })),
                Path::new("fluid_particles.vert"),
                Some(Path::new("sphere_particles.frag")),
                HdrBackbuffer::FORMAT,
                Some(Screen::FORMAT_DEPTH),
            ),
        );
        ParticleRenderer { render_pipeline }
    }

    pub fn draw<'a>(&'a self, rpass: &mut wgpu::RenderPass<'a>, pipeline_manager: &'a PipelineManager, fluid: &'a HybridFluid) {
        rpass.set_pipeline(pipeline_manager.get_render(&self.render_pipeline));
        rpass.set_bind_group(1, fluid.bind_group_renderer(), &[]);
        rpass.draw(0..4, 0..fluid.num_particles());
    }
}
