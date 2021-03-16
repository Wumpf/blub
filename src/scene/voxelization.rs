use std::{path::PathBuf, rc::Rc};

use crate::scene::SceneModels;
use crate::wgpu_utils::{binding_builder::*, binding_glsl, pipelines::*, shader::ShaderDirectory};

pub struct SceneVoxelization {
    voxelization_pipeline: RenderPipelineHandle,
    bind_group: wgpu::BindGroup,
    volume_view: wgpu::TextureView,
}

impl SceneVoxelization {
    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Uint;

    pub fn new(
        device: &wgpu::Device,
        shader_dir: &ShaderDirectory,
        pipeline_manager: &mut PipelineManager,
        global_bind_group_layout: &wgpu::BindGroupLayout,
        grid_dimension: wgpu::Extent3d,
    ) -> Self {
        let volume = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("voxel volume"),
            size: grid_dimension,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: Self::FORMAT,
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::STORAGE,
        });
        let volume_view = volume.create_view(&Default::default());

        let group_layout = BindGroupLayoutBuilder::new()
            .next_binding_fragment(binding_glsl::uimage3D(Self::FORMAT, wgpu::StorageTextureAccess::ReadWrite))
            .create(device, "BindGroupLayout: Voxelization");

        let bind_group = BindGroupBuilder::new(&group_layout)
            .texture(&volume_view)
            .create(device, "BindGroup: Voxelization");

        let mut desc = RenderPipelineCreationDesc {
            label: "Voxelize Mesh",
            layout: Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Voxelize Mesh Pipeline Layout"),
                bind_group_layouts: &[&global_bind_group_layout, &group_layout.layout],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT,
                    range: 0..4,
                }],
            })),
            vertex: VertexStateCreationDesc {
                shader_relative_path: PathBuf::from("voxelize_mesh.vert"),
                buffers: vec![SceneModels::vertex_buffer_layout_position_only()],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: FragmentStateCreationDesc {
                shader_relative_path: PathBuf::from("voxelize_mesh.frag"),
                targets: Vec::new(),
            },
        };
        desc.primitive.topology = wgpu::PrimitiveTopology::TriangleStrip;
        let voxelization_pipeline = pipeline_manager.create_render_pipeline(device, shader_dir, desc);

        SceneVoxelization {
            voxelization_pipeline,
            bind_group,
            volume_view,
        }
    }

    pub fn texture_view(&self) -> &wgpu::TextureView {
        &self.volume_view
    }

    pub fn update<'a>(&'a self, cpass: &mut wgpu::ComputePass<'a>, pipeline_manager: &'a PipelineManager) {
        cpass.set_bind_group(1, &self.bind_group, &[]);
        cpass.set_pipeline(pipeline_manager.get_compute(&self.voxelization_pipeline));
        // todo
    }
}
