use std::{path::PathBuf, rc::Rc};

use crate::{
    render_output::{hdr_backbuffer::HdrBackbuffer, screen::Screen},
    scene::models::SceneModels,
    wgpu_utils::{pipelines::*, shader::ShaderDirectory},
};

pub struct MeshRenderer {
    render_pipeline: RenderPipelineHandle,
}

impl MeshRenderer {
    pub fn new(
        device: &wgpu::Device,
        shader_dir: &ShaderDirectory,
        pipeline_manager: &mut PipelineManager,
        global_bind_group_layout: &wgpu::BindGroupLayout,
        background_and_lighting_group_layout: &wgpu::BindGroupLayout,
    ) -> MeshRenderer {
        let render_pipeline = pipeline_manager.create_render_pipeline(
            device,
            shader_dir,
            RenderPipelineCreationDesc {
                label: "MeshRenderer",
                layout: Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("MeshRenderer Pipeline Layout"),
                    bind_group_layouts: &[global_bind_group_layout, background_and_lighting_group_layout],
                    push_constant_ranges: &[wgpu::PushConstantRange {
                        stages: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT,
                        range: 0..4,
                    }],
                })),
                vertex: VertexStateCreationDesc {
                    shader_relative_path: PathBuf::from("mesh.vert"),
                    buffers: vec![SceneModels::vertex_buffer_layout()],
                },
                primitive: wgpu::PrimitiveState {
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(depth_state::default_read_write(Screen::FORMAT_DEPTH)),
                multisample: Default::default(),
                fragment: FragmentStateCreationDesc {
                    shader_relative_path: PathBuf::from("mesh.frag"),
                    targets: vec![HdrBackbuffer::FORMAT.into()],
                },
            },
        );
        MeshRenderer { render_pipeline }
    }

    // Render pass is assumed to have the global bindings set
    pub fn draw<'a>(
        &'a self,
        rpass: &mut wgpu::RenderPass<'a>,
        pipeline_manager: &'a PipelineManager,
        background_and_lighting_bind_group: &'a wgpu::BindGroup,
        scene_models: &'a SceneModels,
    ) {
        rpass.set_pipeline(pipeline_manager.get_render(&self.render_pipeline));
        rpass.set_bind_group(1, background_and_lighting_bind_group, &[]);

        rpass.set_index_buffer(scene_models.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        rpass.set_vertex_buffer(0, scene_models.vertex_buffer.slice(..));

        for (i, mesh) in scene_models.meshes.iter().enumerate() {
            rpass.set_push_constants(
                wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT,
                0,
                bytemuck::cast_slice(&[i as u32]),
            );
            rpass.draw_indexed(mesh.index_buffer_range.clone(), mesh.vertex_buffer_range.start as i32, 0..1);
        }
    }
}
