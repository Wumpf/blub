// TODO: Not a particle renderer yet.
// The idea is to have different render backend for the fluid, which one being the particle renderer which renders the fluid as particles (sprites)

use super::camera;
use super::shader::*;
use std::path::Path;

pub struct ParticleRenderer {
    render_pipeline: wgpu::RenderPipeline,
    pipeline_layout: wgpu::PipelineLayout,
    bind_group: wgpu::BindGroup,
}

impl ParticleRenderer {
    pub fn new(device: &wgpu::Device, shader_dir: &ShaderDirectory, ubo_camera: &camera::CameraUniformBuffer) -> ParticleRenderer {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            bindings: &[wgpu::BindGroupLayoutBinding {
                binding: 0,
                visibility: wgpu::ShaderStage::VERTEX,
                ty: wgpu::BindingType::UniformBuffer { dynamic: false },
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&bind_group_layout],
        });
        let render_pipeline = Self::create_pipeline_state(device, &pipeline_layout, shader_dir).unwrap();

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            bindings: &[wgpu::Binding {
                binding: 0,
                resource: wgpu::BindingResource::Buffer {
                    buffer: ubo_camera.buffer(),
                    range: 0..std::mem::size_of::<camera::CameraUniformBufferContent>() as u64,
                },
            }],
        });

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
        let vs_module = shader_dir.load_shader_module(device, Path::new("shader.vert"))?;
        let fs_module = shader_dir.load_shader_module(device, Path::new("shader.frag"))?;

        Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            layout: &pipeline_layout,
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &vs_module,
                entry_point: "main",
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &fs_module,
                entry_point: "main",
            }),
            rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: wgpu::CullMode::None,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
            }),
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            color_states: &[wgpu::ColorStateDescriptor {
                format: super::Application::backbuffer_format(),
                color_blend: wgpu::BlendDescriptor::REPLACE,
                alpha_blend: wgpu::BlendDescriptor::REPLACE,
                write_mask: wgpu::ColorWrite::ALL,
            }],
            depth_stencil_state: None,
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

    pub fn draw(&self, rpass: &mut wgpu::RenderPass) {
        rpass.set_pipeline(&self.render_pipeline);
        rpass.set_bind_group(0, &self.bind_group, &[]);
        rpass.draw(0..3, 0..1);
    }
}
