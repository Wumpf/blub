use crate::render_output::hdr_backbuffer::HdrBackbuffer;
use crate::render_output::screen::Screen;
use crate::wgpu_utils::pipelines::*;
use crate::{
    simulation::HybridFluid,
    wgpu_utils::{
        self,
        binding_builder::{BindGroupBuilder, BindGroupLayoutBuilder, BindGroupLayoutWithDesc},
        binding_glsl,
        shader::*,
    },
};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use wgpu_profiler::{wgpu_profiler, GpuProfiler};

struct ScreenDependentProperties {
    texture_view_fluid_view: [wgpu::TextureView; 2],
    texture_view_fluid_thickness: [wgpu::TextureView; 2],
    backbuffer_copy: wgpu::Texture,
    bind_group_narrow_range_filter: [wgpu::BindGroup; 2],
    bind_group_thickness_filter: [wgpu::BindGroup; 2],
    bind_group_compose: wgpu::BindGroup,
    target_textures_resolution: wgpu::Extent3d,
}

struct ScreenIndependentProperties {
    pipeline_render_particles: RenderPipelineHandle,

    pipeline_narrow_range_filter_1d: ComputePipelineHandle,
    pipeline_narrow_range_filter_2d: ComputePipelineHandle,
    group_layout_narrow_range_filter: BindGroupLayoutWithDesc,

    pipeline_thickness_filter: ComputePipelineHandle,
    group_layout_thickness_filter: BindGroupLayoutWithDesc,

    pipeline_fluid: ComputePipelineHandle,
    group_layout_compose: BindGroupLayoutWithDesc,
}

pub struct ScreenSpaceFluid {
    screen_independent: ScreenIndependentProperties,
    screen_dependent: ScreenDependentProperties,
}

impl ScreenSpaceFluid {
    const FORMAT_FLUID_DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
    const FORMAT_FLUID_THICKNESS: wgpu::TextureFormat = wgpu::TextureFormat::R16Float; // TODO: Smaller?

    pub fn new(
        device: &wgpu::Device,
        shader_dir: &ShaderDirectory,
        pipeline_manager: &mut PipelineManager,
        global_bind_group_layout: &wgpu::BindGroupLayout,
        fluid_renderer_group_layout: &wgpu::BindGroupLayout,
        background_and_lighting_group_layout: &wgpu::BindGroupLayout,
        backbuffer: &HdrBackbuffer,
    ) -> ScreenSpaceFluid {
        let group_layout_narrow_range_filter = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::image2D(Self::FORMAT_FLUID_DEPTH, wgpu::StorageTextureAccess::WriteOnly)) // Fluid depth target
            .next_binding_compute(binding_glsl::texture2D()) // Fluid depth source
            .create(device, "BindGroupLayout: Narrow Range Filter");
        let group_layout_thickness_filter = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::image2D(Self::FORMAT_FLUID_THICKNESS, wgpu::StorageTextureAccess::WriteOnly)) // Fluid depth target
            .next_binding_compute(binding_glsl::texture2D()) // Fluid depth source
            .create(device, "BindGroupLayout: Narrow Range Filter");

        let group_layout_compose = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::texture2D()) // Fluid depth
            .next_binding_compute(binding_glsl::texture2D()) // Fluid thickness
            .next_binding_compute(binding_glsl::texture2D()) // HdrBackbuffer copy for reading
            .next_binding_compute(binding_glsl::image2D(HdrBackbuffer::FORMAT, wgpu::StorageTextureAccess::ReadWrite)) // hdr backbuffer, target
            .create(device, "BindGroupLayout: SSFluid, Final fluid/Compose");

        let pipeline_render_particles = pipeline_manager.create_render_pipeline(
            device,
            shader_dir,
            RenderPipelineCreationDesc {
                label: "ScreenspaceFluid: Render Particles",
                layout: Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Render Particles for SS Fluid Pipeline Layout"),
                    bind_group_layouts: &[&global_bind_group_layout, &fluid_renderer_group_layout],
                    push_constant_ranges: &[],
                })),

                vertex: VertexStateCreationDesc {
                    shader_relative_path: PathBuf::from("screenspace_fluid/particles.vert"),
                    buffers: Vec::new(),
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: Screen::FORMAT_DEPTH,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                fragment: FragmentStateCreationDesc {
                    shader_relative_path: PathBuf::from("screenspace_fluid/particles.frag"),
                    targets: vec![
                        wgpu::ColorTargetState {
                            blend: Some(wgpu::BlendState {
                                color: wgpu::BlendComponent {
                                    src_factor: wgpu::BlendFactor::One,
                                    dst_factor: wgpu::BlendFactor::One,
                                    operation: wgpu::BlendOperation::Min,
                                },
                                alpha: wgpu::BlendComponent::REPLACE,
                            }),
                            ..Self::FORMAT_FLUID_DEPTH.into()
                        },
                        wgpu::ColorTargetState {
                            blend: Some(wgpu::BlendState {
                                color: wgpu::BlendComponent {
                                    src_factor: wgpu::BlendFactor::One,
                                    dst_factor: wgpu::BlendFactor::One,
                                    operation: wgpu::BlendOperation::Add,
                                },
                                alpha: wgpu::BlendComponent::REPLACE,
                            }),
                            ..Self::FORMAT_FLUID_THICKNESS.into()
                        },
                    ],
                },
            },
        );

        // Use same push constant range for all compute pipelines to improve internal Vulkan pipeline compatibility.
        let push_constant_ranges = &[wgpu::PushConstantRange {
            stages: wgpu::ShaderStage::COMPUTE,
            range: 0..4,
        }];

        let layout_narrow_range_filter = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Narrow Range Filter Pipeline Layout"),
            bind_group_layouts: &[
                &global_bind_group_layout,
                &fluid_renderer_group_layout,
                &group_layout_narrow_range_filter.layout,
            ],
            push_constant_ranges,
        }));
        let pipeline_narrow_range_filter_1d = pipeline_manager.create_compute_pipeline(
            device,
            shader_dir,
            ComputePipelineCreationDesc::new(
                "ScreenspaceFluid: NarrowRane 1D",
                layout_narrow_range_filter.clone(),
                Path::new("screenspace_fluid/narrow_range_filter_1d.comp"),
            ),
        );
        let pipeline_narrow_range_filter_2d = pipeline_manager.create_compute_pipeline(
            device,
            shader_dir,
            ComputePipelineCreationDesc::new(
                "ScreenspaceFluid: NarrowRane 2D",
                layout_narrow_range_filter.clone(),
                Path::new("screenspace_fluid/narrow_range_filter_2d.comp"),
            ),
        );

        let layout_thickness_filter = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Thickness Filter Pipeline Layout"),
            bind_group_layouts: &[
                &global_bind_group_layout,
                &fluid_renderer_group_layout,
                &group_layout_thickness_filter.layout,
            ],
            push_constant_ranges,
        }));
        let pipeline_thickness_filter = pipeline_manager.create_compute_pipeline(
            device,
            shader_dir,
            ComputePipelineCreationDesc::new(
                "ScreenspaceFluid: Thickness filter",
                layout_thickness_filter.clone(),
                Path::new("screenspace_fluid/thickness_filter.comp"),
            ),
        );

        let pipeline_fluid = pipeline_manager.create_compute_pipeline(
            device,
            shader_dir,
            ComputePipelineCreationDesc::new(
                "ScreenspaceFluid: Fluid/compose",
                Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Fluid Render Pipeline Layout"),
                    bind_group_layouts: &[
                        &global_bind_group_layout,
                        &background_and_lighting_group_layout,
                        &group_layout_compose.layout,
                    ],
                    push_constant_ranges,
                })),
                Path::new("screenspace_fluid/fluid_render.comp"),
            ),
        );

        let screen_independent = ScreenIndependentProperties {
            pipeline_render_particles,

            pipeline_narrow_range_filter_1d,
            pipeline_narrow_range_filter_2d,
            group_layout_narrow_range_filter,

            pipeline_thickness_filter,
            group_layout_thickness_filter,

            pipeline_fluid,
            group_layout_compose,
        };

        let screen_dependent = Self::create_screen_dependent_properties(&screen_independent, device, backbuffer);

        ScreenSpaceFluid {
            screen_dependent,
            screen_independent,
        }
    }

    fn create_screen_dependent_properties(
        screen_independent: &ScreenIndependentProperties,
        device: &wgpu::Device,
        backbuffer: &HdrBackbuffer,
    ) -> ScreenDependentProperties {
        let target_textures_resolution = wgpu::Extent3d {
            width: backbuffer.resolution().width,
            height: backbuffer.resolution().height,
            depth_or_array_layers: 1,
        };
        let texture_fluid_depth = [
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Texture: Fluid Depth 1 (render target)"),
                size: target_textures_resolution,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: Self::FORMAT_FLUID_DEPTH,
                usage: wgpu::TextureUsage::RENDER_ATTACHMENT | wgpu::TextureUsage::STORAGE | wgpu::TextureUsage::SAMPLED,
            }),
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Texture: Fluid Depth 2 (blur target)"),
                size: target_textures_resolution,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: Self::FORMAT_FLUID_DEPTH,
                usage: wgpu::TextureUsage::RENDER_ATTACHMENT | wgpu::TextureUsage::STORAGE | wgpu::TextureUsage::SAMPLED,
            }),
        ];
        let backbuffer_copy = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Texture: HdrBackbuffer Copy for Refraction"),
            size: wgpu::Extent3d {
                width: backbuffer.resolution().width,
                height: backbuffer.resolution().height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HdrBackbuffer::FORMAT,
            usage: wgpu::TextureUsage::COPY_DST | wgpu::TextureUsage::SAMPLED,
        });
        let texture_view_backbuffer_copy = backbuffer_copy.create_view(&Default::default());
        let texture_view_fluid_view = [
            texture_fluid_depth[0].create_view(&Default::default()),
            texture_fluid_depth[1].create_view(&Default::default()),
        ];

        let texture_fluid_thickness = [
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Texture: Fluid Thickness 1 (render target)"),
                size: target_textures_resolution,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: Self::FORMAT_FLUID_THICKNESS,
                usage: wgpu::TextureUsage::RENDER_ATTACHMENT | wgpu::TextureUsage::STORAGE | wgpu::TextureUsage::SAMPLED,
            }),
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Texture: Fluid Thickness 2 (blur target)"),
                size: target_textures_resolution,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: Self::FORMAT_FLUID_THICKNESS,
                usage: wgpu::TextureUsage::RENDER_ATTACHMENT | wgpu::TextureUsage::STORAGE | wgpu::TextureUsage::SAMPLED,
            }),
        ];
        let texture_view_fluid_thickness = [
            texture_fluid_thickness[0].create_view(&Default::default()),
            texture_fluid_thickness[1].create_view(&Default::default()),
        ];

        let bind_group_narrow_range_filter = [
            BindGroupBuilder::new(&screen_independent.group_layout_narrow_range_filter)
                .texture(&texture_view_fluid_view[1])
                .texture(&texture_view_fluid_view[0])
                .create(device, "BindGroup: Narrow Range filter 1"),
            BindGroupBuilder::new(&screen_independent.group_layout_narrow_range_filter)
                .texture(&texture_view_fluid_view[0])
                .texture(&texture_view_fluid_view[1])
                .create(device, "BindGroup: Narrow Range filter 2"),
        ];
        let bind_group_thickness_filter = [
            BindGroupBuilder::new(&screen_independent.group_layout_thickness_filter)
                .texture(&texture_view_fluid_thickness[1])
                .texture(&texture_view_fluid_thickness[0])
                .create(device, "BindGroup: Thickness Filter 1"),
            BindGroupBuilder::new(&screen_independent.group_layout_thickness_filter)
                .texture(&texture_view_fluid_thickness[0])
                .texture(&texture_view_fluid_thickness[1])
                .create(device, "BindGroup: Thickness Filter 2"),
        ];
        let bind_group_compose = BindGroupBuilder::new(&screen_independent.group_layout_compose)
            .texture(&texture_view_fluid_view[1])
            .texture(&texture_view_fluid_thickness[0])
            .texture(&texture_view_backbuffer_copy)
            .texture(&backbuffer.texture_view())
            .create(device, "BindGroup: SSFluid, Final Compose");

        ScreenDependentProperties {
            texture_view_fluid_view,
            texture_view_fluid_thickness,
            backbuffer_copy,
            target_textures_resolution,
            bind_group_narrow_range_filter,
            bind_group_thickness_filter,
            bind_group_compose,
        }
    }

    pub fn on_window_resize(&mut self, device: &wgpu::Device, backbuffer: &HdrBackbuffer) {
        self.screen_dependent = Self::create_screen_dependent_properties(&self.screen_independent, device, backbuffer);
    }

    pub fn draw<'a>(
        &'a self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        profiler: &mut GpuProfiler,
        pipeline_manager: &'a PipelineManager,
        depthbuffer: &wgpu::TextureView,
        global_bind_group: &wgpu::BindGroup,
        background_and_lighting_bind_group: &wgpu::BindGroup,
        fluid: &HybridFluid,
        backbuffer: &HdrBackbuffer,
    ) {
        // Set some depth value that is beyond the far plane. (could do infinity, but don't trust this is passed down correctly)
        let depth_clear_color = wgpu::Color {
            r: 999999.0,
            g: 999999.0,
            b: 999999.0,
            a: 999999.0,
        };

        encoder.copy_texture_to_texture(
            wgpu::ImageCopyTexture {
                texture: backbuffer.texture(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::ImageCopyTexture {
                texture: &self.screen_dependent.backbuffer_copy,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::Extent3d {
                width: backbuffer.resolution().width,
                height: backbuffer.resolution().height,
                depth_or_array_layers: 1,
            },
        );

        wgpu_profiler!("particles", profiler, encoder, device, {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("particles"),
                color_attachments: &[
                    wgpu::RenderPassColorAttachment {
                        view: &self.screen_dependent.texture_view_fluid_view[0],
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(depth_clear_color),
                            store: true,
                        },
                    },
                    wgpu::RenderPassColorAttachment {
                        view: &self.screen_dependent.texture_view_fluid_thickness[0],
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: true,
                        },
                    },
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depthbuffer,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: false,
                    }),
                    stencil_ops: None,
                }),
            });
            rpass.set_bind_group(0, &global_bind_group, &[]);
            rpass.set_bind_group(1, fluid.bind_group_renderer(), &[]);
            rpass.set_pipeline(pipeline_manager.get_render(&self.screen_independent.pipeline_render_particles));
            rpass.draw(0..4, 0..fluid.num_particles());
        });

        wgpu_profiler!("clear intermediate blur targets", profiler, encoder, device, {
            {
                encoder
                    .begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("clear secondary water depth texture"),
                        color_attachments: &[wgpu::RenderPassColorAttachment {
                            view: &self.screen_dependent.texture_view_fluid_view[1],
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(depth_clear_color),
                                store: true,
                            },
                        }],
                        depth_stencil_attachment: None,
                    })
                    .insert_debug_marker("clear secondary water depth texture");
            }
            {
                encoder
                    .begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("clear secondary water thickness texture"),
                        color_attachments: &[wgpu::RenderPassColorAttachment {
                            view: &self.screen_dependent.texture_view_fluid_thickness[1],
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: true,
                            },
                        }],
                        depth_stencil_attachment: None,
                    })
                    .insert_debug_marker("clear secondary water thickness texture");
            }
        });

        wgpu_profiler!("fluid filters & render", profiler, encoder, device, {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fluid filters & render"),
            });
            cpass.set_bind_group(0, &global_bind_group, &[]);
            cpass.set_bind_group(1, fluid.bind_group_renderer(), &[]);

            const LOCAL_SIZE_FILTER_1D_X: wgpu::Extent3d = wgpu::Extent3d {
                width: 64,
                height: 1,
                depth_or_array_layers: 1,
            };
            const LOCAL_SIZE_FILTER_1D_Y: wgpu::Extent3d = wgpu::Extent3d {
                width: 1, // xy not actually swizzled in the local_size shader definition but we treat it like that
                height: 64,
                depth_or_array_layers: 1,
            };
            let work_group_filter_1d_x = wgpu_utils::compute_group_size(self.screen_dependent.target_textures_resolution, LOCAL_SIZE_FILTER_1D_X);
            let work_group_filter_1d_y = wgpu_utils::compute_group_size(self.screen_dependent.target_textures_resolution, LOCAL_SIZE_FILTER_1D_Y);

            wgpu_profiler!("depth filter", profiler, &mut cpass, device, {
                wgpu_profiler!("filter 1D", profiler, &mut cpass, device, {
                    cpass.set_pipeline(pipeline_manager.get_compute(&self.screen_independent.pipeline_narrow_range_filter_1d));

                    // Filter Y
                    cpass.set_bind_group(2, &self.screen_dependent.bind_group_narrow_range_filter[0], &[]);
                    cpass.set_push_constants(0, &bytemuck::bytes_of(&[1 as u32]));
                    cpass.dispatch(
                        work_group_filter_1d_y.width,
                        work_group_filter_1d_y.height,
                        work_group_filter_1d_y.depth_or_array_layers,
                    );
                    // Filter X - note that since filter is not really separable, order makes a difference. Found this order visually more pleasing.
                    cpass.set_bind_group(2, &self.screen_dependent.bind_group_narrow_range_filter[1], &[]);
                    cpass.set_push_constants(0, &bytemuck::bytes_of(&[0 as u32]));
                    cpass.dispatch(
                        work_group_filter_1d_x.width,
                        work_group_filter_1d_x.height,
                        work_group_filter_1d_x.depth_or_array_layers,
                    );
                });
                wgpu_profiler!("filter 2D", profiler, &mut cpass, device, {
                    cpass.set_pipeline(pipeline_manager.get_compute(&self.screen_independent.pipeline_narrow_range_filter_2d));
                    cpass.set_bind_group(2, &self.screen_dependent.bind_group_narrow_range_filter[0], &[]);
                    const LOCAL_SIZE_FILTER_2D: wgpu::Extent3d = wgpu::Extent3d {
                        width: 16,
                        height: 16,
                        depth_or_array_layers: 1,
                    };
                    let work_group = wgpu_utils::compute_group_size(self.screen_dependent.target_textures_resolution, LOCAL_SIZE_FILTER_2D);
                    cpass.dispatch(work_group.width, work_group.height, work_group.depth_or_array_layers);
                });
            });
            wgpu_profiler!("thickness filter", profiler, &mut cpass, device, {
                cpass.set_pipeline(pipeline_manager.get_compute(&self.screen_independent.pipeline_thickness_filter));

                // Filter Y
                cpass.set_bind_group(2, &self.screen_dependent.bind_group_thickness_filter[0], &[]);
                cpass.set_push_constants(0, &bytemuck::bytes_of(&[1 as u32]));
                cpass.dispatch(
                    work_group_filter_1d_y.width,
                    work_group_filter_1d_y.height,
                    work_group_filter_1d_y.depth_or_array_layers,
                );
                // Filter X
                cpass.set_bind_group(2, &self.screen_dependent.bind_group_thickness_filter[1], &[]);
                cpass.set_push_constants(0, &bytemuck::bytes_of(&[0 as u32]));
                cpass.dispatch(
                    work_group_filter_1d_x.width,
                    work_group_filter_1d_x.height,
                    work_group_filter_1d_x.depth_or_array_layers,
                );
            });

            wgpu_profiler!("compose & render", profiler, &mut cpass, device, {
                const LOCAL_SIZE_COMPOSE: wgpu::Extent3d = wgpu::Extent3d {
                    width: 32,
                    height: 32,
                    depth_or_array_layers: 1,
                };

                cpass.set_bind_group(1, background_and_lighting_bind_group, &[]);
                cpass.set_bind_group(2, &self.screen_dependent.bind_group_compose, &[]);
                cpass.set_pipeline(pipeline_manager.get_compute(&self.screen_independent.pipeline_fluid));
                let work_group = wgpu_utils::compute_group_size(self.screen_dependent.target_textures_resolution, LOCAL_SIZE_COMPOSE);
                cpass.dispatch(work_group.width, work_group.height, work_group.depth_or_array_layers);
            });
        });
    }
}
