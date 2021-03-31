use std::{path::Path, rc::Rc};

use crate::wgpu_utils::{binding_builder::*, binding_glsl, pipelines::*, shader::ShaderDirectory};
use crate::{
    render_output::{hdr_backbuffer::HdrBackbuffer, screen::Screen},
    scene::Scene,
};

pub struct VoxelRenderer {
    pipeline: RenderPipelineHandle,
    group_layout: BindGroupLayoutWithDesc,
    bind_group: Option<wgpu::BindGroup>,
}

impl VoxelRenderer {
    pub fn new(
        device: &wgpu::Device,
        shader_dir: &ShaderDirectory,
        pipeline_manager: &mut PipelineManager,
        global_bind_group_layout: &wgpu::BindGroupLayout,
        background_and_lighting_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let group_layout = BindGroupLayoutBuilder::new()
            .next_binding(wgpu::ShaderStage::VERTEX_FRAGMENT, binding_glsl::texture3D())
            .create(device, "BindGroupLayout: Voxel Renderer");

        let mut desc = RenderPipelineCreationDesc::new(
            "Visualize Voxels",
            Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Visualize Voxels Pipeline Layout"),
                bind_group_layouts: &[&global_bind_group_layout, background_and_lighting_group_layout, &group_layout.layout],
                push_constant_ranges: &[],
            })),
            Path::new("volume_visualization/voxel_visualization.vert"),
            Path::new("volume_visualization/voxel_visualization.frag"),
            HdrBackbuffer::FORMAT,
            Some(Screen::FORMAT_DEPTH),
        );
        desc.primitive.topology = wgpu::PrimitiveTopology::TriangleStrip;
        let pipeline = pipeline_manager.create_render_pipeline(device, shader_dir, desc);

        VoxelRenderer {
            pipeline,
            group_layout,
            bind_group: None,
        }
    }

    pub fn on_new_scene(&mut self, device: &wgpu::Device, scene: &Scene) {
        self.bind_group = Some(
            BindGroupBuilder::new(&self.group_layout)
                .texture(scene.voxelization.texture_view())
                .create(device, "BindGroup: Voxel Renderer"),
        );
    }

    pub fn draw<'a>(
        &'a self,
        rpass: &mut wgpu::RenderPass<'a>,
        pipeline_manager: &'a PipelineManager,
        background_and_lighting_bind_group: &'a wgpu::BindGroup,
        grid_dimension: &cgmath::Point3<u32>,
    ) {
        let bind_group = match self.bind_group.as_ref() {
            Some(bind_group) => bind_group,
            None => {
                return;
            }
        };

        rpass.set_pipeline(pipeline_manager.get_render(&self.pipeline));
        rpass.set_bind_group(1, background_and_lighting_bind_group, &[]);
        rpass.set_bind_group(2, bind_group, &[]);

        // this is heavy, but fine for debug viz..
        rpass.draw(0..14, 0..(grid_dimension.x * grid_dimension.y * grid_dimension.z));
    }
}
