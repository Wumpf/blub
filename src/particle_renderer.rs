use crate::hybrid_fluid::*;
use crate::wgpu_utils::shader::*;
use std::path::Path;

pub struct ParticleRenderer {
    render_pipeline: wgpu::RenderPipeline,
    pipeline_layout: wgpu::PipelineLayout,
}

impl ParticleRenderer {
    pub fn new(
        device: &wgpu::Device,
        shader_dir: &ShaderDirectory,
        per_frame_bind_group_layout: &wgpu::BindGroupLayout,
        fluid_renderer_group_layout: &wgpu::BindGroupLayout,
    ) -> ParticleRenderer {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&per_frame_bind_group_layout, &fluid_renderer_group_layout],
        });
        let render_pipeline = Self::create_pipeline_state(device, &pipeline_layout, shader_dir).unwrap();

        ParticleRenderer {
            render_pipeline,
            pipeline_layout,
        }
    }

    fn create_pipeline_state(
        device: &wgpu::Device,
        pipeline_layout: &wgpu::PipelineLayout,
        shader_dir: &ShaderDirectory,
    ) -> Result<wgpu::RenderPipeline, ()> {
        let vs_module = shader_dir.load_shader_module(device, Path::new("sphere_particles.vert"))?;
        let fs_module = shader_dir.load_shader_module(device, Path::new("sphere_particles.frag"))?;

        Ok(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            layout: &pipeline_layout,
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &vs_module,
                entry_point: SHADER_ENTRY_POINT_NAME,
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &fs_module,
                entry_point: SHADER_ENTRY_POINT_NAME,
            }),
            rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: wgpu::CullMode::None,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
            }),
            primitive_topology: wgpu::PrimitiveTopology::TriangleStrip,
            color_states: &[wgpu::ColorStateDescriptor {
                format: super::Screen::FORMAT_BACKBUFFER,
                color_blend: wgpu::BlendDescriptor::REPLACE,
                alpha_blend: wgpu::BlendDescriptor::REPLACE,
                write_mask: wgpu::ColorWrite::ALL,
            }],
            depth_stencil_state: Some(wgpu::DepthStencilStateDescriptor {
                format: super::Screen::FORMAT_DEPTH,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil_front: wgpu::StencilStateFaceDescriptor::IGNORE,
                stencil_back: wgpu::StencilStateFaceDescriptor::IGNORE,
                stencil_read_mask: 0,
                stencil_write_mask: 0,
            }),
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: wgpu::IndexFormat::Uint16,
                vertex_buffers: &[],
            },

            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        }))
    }

    pub fn try_reload_shaders(&mut self, device: &wgpu::Device, shader_dir: &ShaderDirectory) {
        if let Ok(render_pipeline) = Self::create_pipeline_state(device, &self.pipeline_layout, shader_dir) {
            self.render_pipeline = render_pipeline;
        }
    }

    pub fn draw<'a>(&'a self, rpass: &mut wgpu::RenderPass<'a>, fluid: &'a HybridFluid) {
        rpass.set_pipeline(&self.render_pipeline);
        rpass.set_bind_group(1, fluid.bind_group_renderer(), &[]);
        rpass.draw(0..4, 0..fluid.num_particles());
    }
}
