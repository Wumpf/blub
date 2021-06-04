use std::rc::Rc;

use crate::scene::SceneModels;
use crate::wgpu_utils::{binding_builder::*, binding_glsl, pipelines::*, shader::ShaderDirectory};

pub struct SceneVoxelization {
    pipeline_conservative_hull: RenderPipelineHandle,
    bind_group: wgpu::BindGroup,
    volume: wgpu::Texture,
    volume_view: wgpu::TextureView,

    dummy_render_target: wgpu::TextureView,
    viewport_extent: u32,
}

impl SceneVoxelization {
    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

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
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::STORAGE | wgpu::TextureUsage::COPY_DST,
        });
        let volume_view = volume.create_view(&Default::default());

        let group_layout = BindGroupLayoutBuilder::new()
            .next_binding(
                wgpu::ShaderStage::COMPUTE | wgpu::ShaderStage::FRAGMENT,
                binding_glsl::image3D(Self::FORMAT, wgpu::StorageTextureAccess::WriteOnly),
            )
            .create(device, "BindGroupLayout: Voxelization");

        let bind_group = BindGroupBuilder::new(&group_layout)
            .texture(&volume_view)
            .create(device, "BindGroup: Voxelization");

        let pipeline_conservative_hull = pipeline_manager.create_render_pipeline(
            device,
            shader_dir,
            RenderPipelineCreationDesc {
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
                    shader_relative_path: "voxelize/conservative_hull.vert".into(),
                    buffers: Vec::new(),
                },
                primitive: wgpu::PrimitiveState {
                    cull_mode: None,
                    conservative: true,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                // Needed until https://github.com/gpuweb/gpuweb/issues/503 is resolved
                fragment: FragmentStateCreationDesc {
                    shader_relative_path: "voxelize/conservative_hull.frag".into(),
                    targets: vec![wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8UnormSrgb,
                        blend: None,
                        write_mask: wgpu::ColorWrite::empty(),
                    }],
                },
            },
        );

        let viewport_extent = grid_dimension.width.max(grid_dimension.height).max(grid_dimension.depth_or_array_layers);

        // Needed until https://github.com/gpuweb/gpuweb/issues/503 is resolved
        let dummy_render_target = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("dummy render target"),
                size: wgpu::Extent3d {
                    width: viewport_extent,
                    height: viewport_extent,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsage::RENDER_ATTACHMENT,
            })
            .create_view(&Default::default());

        SceneVoxelization {
            pipeline_conservative_hull,
            bind_group,
            volume,
            volume_view,

            viewport_extent,
            dummy_render_target,
        }
    }

    pub fn texture_view(&self) -> &wgpu::TextureView {
        &self.volume_view
    }

    pub fn update(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline_manager: &PipelineManager,
        global_bind_group: &wgpu::BindGroup,
        scene_models: &SceneModels,
    ) {
        encoder.clear_texture(&self.volume, &Default::default());

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Voxelize"),
            // Needed until https://github.com/gpuweb/gpuweb/issues/503 is resolved
            color_attachments: &[wgpu::RenderPassColorAttachment {
                view: &self.dummy_render_target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: false,
                },
            }],
            depth_stencil_attachment: None,
        });

        rpass.set_viewport(0.0, 0.0, self.viewport_extent as f32, self.viewport_extent as f32, 0.0, 1.0);
        rpass.set_pipeline(pipeline_manager.get_render(&self.pipeline_conservative_hull));
        rpass.set_bind_group(0, &global_bind_group, &[]);
        rpass.set_bind_group(1, &self.bind_group, &[]);

        // Use programmable vertex fetching since for every triangle we want to decide independently which direction to use for rendering.
        // (i.e. we may need to duplicate vertices that are otherwise shared with triangles)

        for (i, mesh) in scene_models.meshes.iter().enumerate() {
            rpass.set_push_constants(
                wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT,
                0,
                bytemuck::cast_slice(&[i as u32]),
            );
            rpass.draw(mesh.index_buffer_range.clone(), 0..1);
        }
    }
}
