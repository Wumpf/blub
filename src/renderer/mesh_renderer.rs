use std::{path::PathBuf, rc::Rc};

use crate::{
    render_output::{hdr_backbuffer::HdrBackbuffer, screen::Screen},
    scene_models::*,
    wgpu_utils::{binding_builder::*, binding_glsl, pipelines::*, shader::ShaderDirectory},
};

pub struct MeshRenderer {
    render_pipeline: RenderPipelineHandle,
    bind_group_layout: BindGroupLayoutWithDesc,
    // Needs to change with scene.
    bind_group: Option<wgpu::BindGroup>,
}

const VERTEX_SIZE: wgpu::BufferAddress = std::mem::size_of::<MeshVertex>() as wgpu::BufferAddress;

impl MeshRenderer {
    pub fn new(
        device: &wgpu::Device,
        shader_dir: &ShaderDirectory,
        pipeline_manager: &mut PipelineManager,
        per_frame_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> MeshRenderer {
        let bind_group_layout = BindGroupLayoutBuilder::new()
            .next_binding(wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT, binding_glsl::buffer(true))
            .create(device, "BindGroupLayout: Transfer velocity from Particles to Volume(s)");

        let render_pipeline = pipeline_manager.create_render_pipeline(
            device,
            shader_dir,
            RenderPipelineCreationDesc {
                label: "MeshRenderer",
                layout: Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("MeshRenderer Pipeline Layout"),
                    bind_group_layouts: &[&per_frame_bind_group_layout, &bind_group_layout.layout],
                    push_constant_ranges: &[wgpu::PushConstantRange {
                        stages: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT,
                        range: 0..4,
                    }],
                })),
                vertex_shader_relative_path: PathBuf::from("mesh.vert"),
                fragment_shader_relative_path: Some(PathBuf::from("mesh.frag")),
                rasterization_state: Some(rasterization_state::culling_back()),
                primitive_topology: wgpu::PrimitiveTopology::TriangleList,
                color_states: vec![color_state::write_all(HdrBackbuffer::FORMAT)],
                depth_stencil_state: Some(depth_state::default_read_write(Screen::FORMAT_DEPTH)),
                vertex_state: wgpu::VertexStateDescriptor {
                    index_format: wgpu::IndexFormat::Uint32,
                    vertex_buffers: &[wgpu::VertexBufferDescriptor {
                        stride: VERTEX_SIZE,
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
                            wgpu::VertexAttributeDescriptor {
                                format: wgpu::VertexFormat::Float2,
                                offset: 4 * 6,
                                shader_location: 2,
                            },
                        ],
                    }],
                },
                sample_count: 1,
                sample_mask: !0,
                alpha_to_coverage_enabled: false,
            },
        );
        MeshRenderer {
            bind_group_layout,
            render_pipeline,
            bind_group: None,
        }
    }

    pub fn on_new_scene(&mut self, device: &wgpu::Device, scene_models: &SceneModels) {
        self.bind_group = Some(
            BindGroupBuilder::new(&self.bind_group_layout)
                .resource(scene_models.mesh_desc_buffer.as_entire_binding())
                .create(device, "BindGroup: MeshRenderer"),
        );
    }

    pub fn draw<'a>(&'a self, rpass: &mut wgpu::RenderPass<'a>, pipeline_manager: &'a PipelineManager, scene_models: &'a SceneModels) {
        rpass.set_pipeline(pipeline_manager.get_render(&self.render_pipeline));
        let bind_group = self.bind_group.as_ref().expect("No bind group for mesh renderer, no scene loaded?");
        rpass.set_bind_group(1, bind_group, &[]);
        rpass.set_index_buffer(scene_models.index_buffer.slice(..));
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
