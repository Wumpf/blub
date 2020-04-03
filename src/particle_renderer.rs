// TODO: Not a particle renderer yet.
// The idea is to have different render backend for the fluid, which one being the particle renderer which renders the fluid as particles (sprites)

use super::camera::CameraUniformBuffer;
use super::hybrid_fluid::*;
use super::shader::*;
use crate::wgpu_utils::*;
use std::path::Path;

pub struct ParticleRenderer {
    render_pipeline: wgpu::RenderPipeline,
    pipeline_layout: wgpu::PipelineLayout,
    bind_group: wgpu::BindGroup,
}

impl ParticleRenderer {
    pub fn new(
        device: &wgpu::Device,
        shader_dir: &ShaderDirectory,
        ubo_camera: &CameraUniformBuffer,
        hybrid_fluid: &HybridFluid,
    ) -> ParticleRenderer {
        let bind_group_layout = BindGroupLayoutBuilder::new()
            .next_binding_rendering(wgpu::BindingType::UniformBuffer { dynamic: false })
            .next_binding_vertex(bindingtype_storagebuffer_readonly())
            .create(device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&bind_group_layout.layout],
        });
        let render_pipeline = Self::create_pipeline_state(device, &pipeline_layout, shader_dir).unwrap();

        let bind_group = BindGroupBuilder::new(&bind_group_layout)
            .resource(ubo_camera.binding_resource())
            .resource(hybrid_fluid.particle_binding_resource())
            .create(device);

        ParticleRenderer {
            render_pipeline,
            pipeline_layout,
            bind_group,
        }
    }

    fn create_pipeline_state(
        device: &wgpu::Device,
        pipeline_layout: &wgpu::PipelineLayout,
        shader_dir: &ShaderDirectory,
    ) -> Option<wgpu::RenderPipeline> {
        let vs_module = shader_dir.load_shader_module(device, Path::new("sphere_particles.vert"))?;
        let fs_module = shader_dir.load_shader_module(device, Path::new("sphere_particles.frag"))?;

        Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
            index_format: wgpu::IndexFormat::Uint16,
            vertex_buffers: &[],
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        }))
    }

    pub fn try_reload_shaders(&mut self, device: &wgpu::Device, shader_dir: &ShaderDirectory) {
        if let Some(render_pipeline) = Self::create_pipeline_state(device, &self.pipeline_layout, shader_dir) {
            self.render_pipeline = render_pipeline;
        }
    }

    pub fn draw(&self, rpass: &mut wgpu::RenderPass, num_particles: u64) {
        rpass.set_pipeline(&self.render_pipeline);
        rpass.set_bind_group(0, &self.bind_group, &[]);
        rpass.draw(0..4, 0..num_particles as u32);
    }
}
