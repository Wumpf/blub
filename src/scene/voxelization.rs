use std::{path::PathBuf, rc::Rc};

use crate::scene::SceneModels;
use crate::wgpu_utils::{binding_builder::*, binding_glsl, pipelines::*, shader::ShaderDirectory};

pub struct SceneVoxelization {
    clear_pipeline: ComputePipelineHandle,
    voxelization_pipeline: RenderPipelineHandle,
    bind_group: wgpu::BindGroup,
    volume_view: wgpu::TextureView,

    dummy_render_target: wgpu::TextureView,
    grid_dimension: wgpu::Extent3d,
    viewport_extent: u32,
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
            .next_binding(
                wgpu::ShaderStage::COMPUTE | wgpu::ShaderStage::FRAGMENT,
                binding_glsl::image3D(Self::FORMAT, wgpu::StorageTextureAccess::ReadWrite),
            )
            .create(device, "BindGroupLayout: Voxelization");

        let bind_group = BindGroupBuilder::new(&group_layout)
            .texture(&volume_view)
            .create(device, "BindGroup: Voxelization");

        let voxelization_pipeline = pipeline_manager.create_render_pipeline(
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
                    shader_relative_path: "voxelize_mesh.vert".into(),
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
                    shader_relative_path: "voxelize_mesh.frag".into(),
                    targets: vec![wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8UnormSrgb,
                        blend: None,
                        write_mask: wgpu::ColorWrite::empty(),
                    }],
                },
            },
        );

        let clear_pipeline = pipeline_manager.create_compute_pipeline(
            device,
            shader_dir,
            ComputePipelineCreationDesc {
                label: "Clear voxelization",
                layout: Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Clear Voxelization Pipeline Layout"),
                    bind_group_layouts: &[&group_layout.layout],
                    push_constant_ranges: &[],
                })),
                compute_shader_relative_path: "voxelize_clear.comp".into(),
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
            voxelization_pipeline,
            clear_pipeline,
            bind_group,
            volume_view,

            grid_dimension,
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
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Clear Voxelization"),
            });
            cpass.set_pipeline(pipeline_manager.get_compute(&self.clear_pipeline));
            cpass.set_bind_group(0, &self.bind_group, &[]);
            cpass.dispatch(
                self.grid_dimension.width / 4,
                self.grid_dimension.height / 4,
                self.grid_dimension.depth_or_array_layers / 4,
            );
        }

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Voxelize"),
            // Needed until https://github.com/gpuweb/gpuweb/issues/503 is resolved
            color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                attachment: &self.dummy_render_target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: false,
                },
            }],
            depth_stencil_attachment: None,
        });

        rpass.set_viewport(0.0, 0.0, self.viewport_extent as f32, self.viewport_extent as f32, 0.0, 1.0);
        rpass.set_pipeline(pipeline_manager.get_render(&self.voxelization_pipeline));
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
