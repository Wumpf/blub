use super::screen::Screen;
use crate::wgpu_utils::{
    binding_builder::{BindGroupBuilder, BindGroupLayoutBuilder},
    binding_glsl,
    pipelines::{color_state, rasterization_state},
    shader::{ShaderDirectory, SHADER_ENTRY_POINT_NAME},
};
use std::path::Path;

pub struct HdrBackbuffer {
    hdr_backbuffer: wgpu::Texture,
    hdr_backbuffer_view: wgpu::TextureView,
    resolution: winit::dpi::PhysicalSize<u32>,

    read_backbuffer_bind_group: wgpu::BindGroup,
    hdr_resolve_pipeline: wgpu::RenderPipeline,
}

impl HdrBackbuffer {
    pub const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

    pub fn new(device: &wgpu::Device, resolution: winit::dpi::PhysicalSize<u32>, shader_dir: &ShaderDirectory) -> Self {
        let size = wgpu::Extent3d {
            width: resolution.width,
            height: resolution.height,
            depth: 1,
        };

        let hdr_backbuffer = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Texture: HdrBackbuffer"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::FORMAT,
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT | wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::STORAGE | wgpu::TextureUsage::COPY_SRC,
        });
        let hdr_backbuffer_view = hdr_backbuffer.create_view(&Default::default());

        let bind_group_layout = BindGroupLayoutBuilder::new()
            .next_binding_fragment(binding_glsl::texture2D())
            .create(device, "BindGroupLayout: Screen, Read Texture");
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("HdrBackbuffer Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout.layout],
            push_constant_ranges: &[],
        });
        let read_backbuffer_bind_group = BindGroupBuilder::new(&bind_group_layout)
            .texture(&hdr_backbuffer_view)
            .create(device, "BindGroup: Read HdrBackbuffer");

        let vs_module = shader_dir.load_shader_module(device, Path::new("screentri.vert")).unwrap();
        let fs_module = shader_dir.load_shader_module(device, Path::new("copy_texture.frag")).unwrap();
        // TODO: Use pipelinemanager. TODO have central, this is duplicated.
        let hdr_resolve_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("HdrBackbuffer: Copy texture"),
            layout: Some(&pipeline_layout),
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &vs_module,
                entry_point: SHADER_ENTRY_POINT_NAME,
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &fs_module,
                entry_point: SHADER_ENTRY_POINT_NAME,
            }),
            rasterization_state: Some(rasterization_state::culling_none()),
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            color_states: &[color_state::write_all(Screen::FORMAT_BACKBUFFER)],
            depth_stencil_state: None,
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: wgpu::IndexFormat::Uint16,
                vertex_buffers: &[],
            },
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        HdrBackbuffer {
            hdr_backbuffer,
            hdr_backbuffer_view: hdr_backbuffer_view,
            resolution,

            read_backbuffer_bind_group,
            hdr_resolve_pipeline,
        }
    }

    pub fn resolution(&self) -> winit::dpi::PhysicalSize<u32> {
        self.resolution
    }

    pub fn texture(&self) -> &wgpu::Texture {
        &self.hdr_backbuffer
    }

    pub fn texture_view(&self) -> &wgpu::TextureView {
        &self.hdr_backbuffer_view
    }

    pub fn tonemap(&self, target: &wgpu::TextureView, encoder: &mut wgpu::CommandEncoder) {
        wgpu_scope!(encoder, "HdrBackbuffer.tonemap");

        // TODO: All this tonemapping does is go from half (linear) to srgb. Do some nice tonemapping here!
        // Note that we can't use a compute shader here since that would require STORAGE usage flag on the final output which we can't do since it's srgb!
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                attachment: &target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: true,
                },
            }],
            depth_stencil_attachment: None,
        });
        render_pass.set_pipeline(&self.hdr_resolve_pipeline);
        render_pass.set_bind_group(0, &self.read_backbuffer_bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}
