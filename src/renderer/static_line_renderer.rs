use crate::wgpu_utils::pipelines::*;
use crate::{
    render_output::{hdr_backbuffer::HdrBackbuffer, screen::Screen},
    wgpu_utils::shader::*,
};
use std::{path::Path, rc::Rc};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct LineVertex {
    pub position: cgmath::Point3<f32>,
    pub color: cgmath::Vector3<f32>,
}
unsafe impl bytemuck::Pod for LineVertex {}
unsafe impl bytemuck::Zeroable for LineVertex {}

impl LineVertex {
    pub fn new(pos: cgmath::Point3<f32>, color: cgmath::Vector3<f32>) -> Self {
        LineVertex { position: pos, color }
    }
}

const LINE_VERTEX_SIZE: usize = std::mem::size_of::<LineVertex>();

pub struct StaticLineRenderer {
    render_pipeline: RenderPipelineHandle,
    vertex_buffer: wgpu::Buffer,

    max_num_lines: usize,
    num_lines: usize,
}

impl StaticLineRenderer {
    pub fn new(
        device: &wgpu::Device,
        shader_dir: &ShaderDirectory,
        pipeline_manager: &mut PipelineManager,
        global_bind_group_layout: &wgpu::BindGroupLayout,
        max_num_lines: usize,
    ) -> Self {
        let mut render_pipeline_desc = RenderPipelineCreationDesc::new(
            "Line Renderer",
            Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Static Line Renderer Pipeline Layout"),
                bind_group_layouts: &[&global_bind_group_layout],
                push_constant_ranges: &[],
            })),
            Path::new("lines.vert"),
            Path::new("vertex_color.frag"),
            HdrBackbuffer::FORMAT,
            Some(Screen::FORMAT_DEPTH),
        );
        render_pipeline_desc.primitive.topology = wgpu::PrimitiveTopology::LineList;
        render_pipeline_desc.vertex.buffers = vec![wgpu::VertexBufferLayout {
            array_stride: LINE_VERTEX_SIZE as wgpu::BufferAddress,
            step_mode: wgpu::InputStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 4 * 3,
                    shader_location: 1,
                },
            ],
        }];

        let render_pipeline = pipeline_manager.create_render_pipeline(device, shader_dir, render_pipeline_desc);

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("StaticLineRenderer VertexBuffer"),
            size: (max_num_lines * LINE_VERTEX_SIZE * 2) as wgpu::BufferAddress,
            usage: wgpu::BufferUsage::VERTEX | wgpu::BufferUsage::COPY_DST,
            mapped_at_creation: false,
        });

        StaticLineRenderer {
            render_pipeline,
            vertex_buffer,

            max_num_lines,
            num_lines: 0,
        }
    }

    pub fn clear_lines(&mut self) {
        self.num_lines = 0;
    }

    pub fn add_lines(&mut self, lines: &[LineVertex], queue: &wgpu::Queue) {
        if lines.len() + self.num_lines > self.max_num_lines {
            error!(
                "Buffer too small to add {} lines. Containing {} right now, maximum is {}",
                lines.len(),
                self.num_lines,
                self.max_num_lines
            );
            return;
        }

        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(lines));
        self.num_lines += lines.len();
    }

    pub fn draw<'a>(&'a self, rpass: &mut wgpu::RenderPass<'a>, pipeline_manager: &'a PipelineManager) {
        rpass.set_pipeline(pipeline_manager.get_render(&self.render_pipeline));
        let num_vertices = self.num_lines * 2;
        rpass.set_vertex_buffer(0, self.vertex_buffer.slice(0..(num_vertices as u64 * LINE_VERTEX_SIZE as u64)));
        rpass.draw(0..(num_vertices as u32), 0..1);
    }
}
