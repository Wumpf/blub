use crate::wgpu_utils::pipelines::*;
use crate::wgpu_utils::shader::*;
use std::path::Path;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct LineVertex {
    pub position: cgmath::Point3<f32>,
    pub color: cgmath::Vector3<f32>,
}

impl LineVertex {
    pub fn new(pos: cgmath::Point3<f32>, color: cgmath::Vector3<f32>) -> Self {
        LineVertex { position: pos, color }
    }
}

const LINE_VERTEX_SIZE: usize = std::mem::size_of::<LineVertex>();

pub struct StaticLineRenderer {
    render_pipeline: wgpu::RenderPipeline,
    pipeline_layout: wgpu::PipelineLayout,
    vertex_buffer: wgpu::Buffer,

    max_num_lines: usize,
    num_lines: usize,
}

impl StaticLineRenderer {
    pub fn new(
        device: &wgpu::Device,
        shader_dir: &ShaderDirectory,
        per_frame_bind_group_layout: &wgpu::BindGroupLayout,
        max_num_lines: usize,
    ) -> Self {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&per_frame_bind_group_layout],
        });
        let render_pipeline = Self::create_pipeline_state(device, &pipeline_layout, shader_dir).unwrap();

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("StaticLineRenderer VertexBuffer"),
            size: (max_num_lines * LINE_VERTEX_SIZE * 2) as wgpu::BufferAddress,
            usage: wgpu::BufferUsage::VERTEX | wgpu::BufferUsage::COPY_DST,
        });

        StaticLineRenderer {
            render_pipeline,
            pipeline_layout,
            vertex_buffer,

            max_num_lines,
            num_lines: 0,
        }
    }

    fn create_pipeline_state(
        device: &wgpu::Device,
        pipeline_layout: &wgpu::PipelineLayout,
        shader_dir: &ShaderDirectory,
    ) -> Result<wgpu::RenderPipeline, ()> {
        let vs_module = shader_dir.load_shader_module(device, Path::new("lines.vert"))?;
        let fs_module = shader_dir.load_shader_module(device, Path::new("lines.frag"))?;

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
            rasterization_state: Some(rasterization_state::culling_none()),
            primitive_topology: wgpu::PrimitiveTopology::LineList,
            color_states: &[color_state::write_all(super::Screen::FORMAT_BACKBUFFER)],
            depth_stencil_state: Some(depth_state::default_read_write(super::Screen::FORMAT_DEPTH)),
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: wgpu::IndexFormat::Uint16,
                vertex_buffers: &[wgpu::VertexBufferDescriptor {
                    stride: LINE_VERTEX_SIZE as wgpu::BufferAddress,
                    step_mode: wgpu::InputStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttributeDescriptor {
                            format: wgpu::VertexFormat::Float3,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttributeDescriptor {
                            format: wgpu::VertexFormat::Float3,
                            offset: 4 * 3,
                            shader_location: 1,
                        },
                    ],
                }],
            },
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        }))
    }

    pub fn clear_lines(&mut self) {
        self.num_lines = 0;
    }

    pub fn add_lines(&mut self, lines: &[LineVertex], device: &wgpu::Device, init_encoder: &mut wgpu::CommandEncoder) {
        if lines.len() + self.num_lines > self.max_num_lines {
            error!(
                "Buffer too small to add {} lines. Containing {} right now, maximum is {}",
                lines.len(),
                self.num_lines,
                self.max_num_lines
            );
            return;
        }

        let new_vertices_size = (lines.len() * LINE_VERTEX_SIZE) as u64;
        let particle_buffer_mapping = device.create_buffer_mapped(&wgpu::BufferDescriptor {
            label: Some("Buffer: StaticLine Update"),
            size: new_vertices_size,
            usage: wgpu::BufferUsage::COPY_SRC,
        });

        unsafe {
            std::ptr::copy_nonoverlapping(
                lines.as_ptr() as *const u8,
                particle_buffer_mapping.data.as_mut_ptr(),
                new_vertices_size as usize,
            );
        }

        init_encoder.copy_buffer_to_buffer(
            &particle_buffer_mapping.finish(),
            0,
            &self.vertex_buffer,
            (self.num_lines * LINE_VERTEX_SIZE) as u64,
            new_vertices_size,
        );

        self.num_lines += lines.len();
    }

    pub fn try_reload_shaders(&mut self, device: &wgpu::Device, shader_dir: &ShaderDirectory) {
        if let Ok(render_pipeline) = Self::create_pipeline_state(device, &self.pipeline_layout, shader_dir) {
            self.render_pipeline = render_pipeline;
        }
    }

    pub fn draw<'a>(&'a self, rpass: &mut wgpu::RenderPass<'a>) {
        rpass.set_pipeline(&self.render_pipeline);
        let num_vertices = self.num_lines * 2;
        rpass.set_vertex_buffer(0, &self.vertex_buffer, 0, (num_vertices * LINE_VERTEX_SIZE) as u64);
        rpass.draw(0..(num_vertices as u32), 0..1);
    }
}
