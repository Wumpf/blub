use crate::{
    render_output::hdr_backbuffer::HdrBackbuffer,
    render_output::screen::Screen,
    wgpu_utils::{binding_builder::*, binding_glsl, pipelines::*, shader::ShaderDirectory},
};
use std::{path::Path, rc::Rc};

pub struct CubemapRenderer {
    pipeline: RenderPipelineHandle,
    bind_group: wgpu::BindGroup,
}

impl CubemapRenderer {
    pub fn new(
        device: &wgpu::Device,
        shader_dir: &ShaderDirectory,
        pipeline_manager: &mut PipelineManager,
        per_frame_bind_group_layout: &wgpu::BindGroupLayout,
        cubemap_view: &wgpu::TextureView,
    ) -> Self {
        let bind_group_layout = BindGroupLayoutBuilder::new()
            .next_binding_fragment(binding_glsl::textureCube())
            .create(device, "BindGroupLayout: CubemapRenderer");

        let bind_group = BindGroupBuilder::new(&bind_group_layout)
            .texture(cubemap_view)
            .create(device, "BindGroup: CubemapRenderer");

        let mut render_pipeline_desc = RenderPipelineCreationDesc::new(
            "Cubemap Renderer",
            Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Cubemap Renderer Pipeline Layout"),
                bind_group_layouts: &[&per_frame_bind_group_layout, &bind_group_layout.layout],
                push_constant_ranges: &[],
            })),
            Path::new("screentri.vert"),
            Some(Path::new("background_cubemap.frag")),
            HdrBackbuffer::FORMAT,
            None,
        );
        render_pipeline_desc.depth_stencil_state = Some(wgpu::DepthStencilStateDescriptor {
            format: Screen::FORMAT_DEPTH,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::Equal,
            stencil: Default::default(),
        });

        CubemapRenderer {
            pipeline: pipeline_manager.create_render_pipeline(device, shader_dir, render_pipeline_desc),
            bind_group,
        }
    }

    pub fn draw<'a>(&'a self, rpass: &mut wgpu::RenderPass<'a>, pipeline_manager: &'a PipelineManager) {
        wgpu_scope!(rpass, "CubemapRenderer.draw");
        rpass.set_bind_group(1, &self.bind_group, &[]);
        rpass.set_pipeline(pipeline_manager.get_render(&self.pipeline));
        rpass.draw(0..3, 0..1);
    }
}
