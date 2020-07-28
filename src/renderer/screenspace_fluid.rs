use crate::hybrid_fluid::*;
use crate::render_output::hdr_backbuffer::HdrBackbuffer;
use crate::render_output::screen::Screen;
use crate::wgpu_utils::pipelines::*;
use crate::wgpu_utils::{
    self,
    binding_builder::{BindGroupBuilder, BindGroupLayoutBuilder, BindGroupLayoutWithDesc},
    binding_glsl,
    shader::*,
};
use std::path::{Path, PathBuf};
use std::rc::Rc;

struct ScreenDependentProperties {
    texture_view_fluid_depth: wgpu::TextureView,
    texture_view_fluid_thickness: wgpu::TextureView,
    bind_group_compose: wgpu::BindGroup,
    target_textures_resolution: wgpu::Extent3d,
}

struct ScreenIndependentProperties {
    pipeline_compose: ComputePipelineHandle,
    pipeline_render_particles: RenderPipelineHandle,
    group_layout_compose: BindGroupLayoutWithDesc,
}

pub struct ScreenSpaceFluid {
    screen_independent: ScreenIndependentProperties,
    screen_dependent: ScreenDependentProperties,
}

impl ScreenSpaceFluid {
    const FORMAT_FLUID_DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::R32Float; // TODO: Smaller?
    const FORMAT_FLUID_THICKNESS: wgpu::TextureFormat = wgpu::TextureFormat::R16Float; // TODO: Smaller?

    pub fn new(
        device: &wgpu::Device,
        shader_dir: &ShaderDirectory,
        pipeline_manager: &mut PipelineManager,
        per_frame_bind_group_layout: &wgpu::BindGroupLayout,
        fluid_renderer_group_layout: &wgpu::BindGroupLayout,
        backbuffer: &HdrBackbuffer,
    ) -> ScreenSpaceFluid {
        let group_layout_compose = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::texture2D()) // Fluid depth
            .next_binding_compute(binding_glsl::texture2D()) // Fluid thickness
            .next_binding_compute(binding_glsl::image2d(HdrBackbuffer::FORMAT, false)) // hdr backbuffer
            .create(device, "BindGroupLayout: SSFluid, Final Compose");

        let pipeline_render_particles = pipeline_manager.create_render_pipeline(
            device,
            shader_dir,
            RenderPipelineCreationDesc {
                layout: Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    bind_group_layouts: &[&per_frame_bind_group_layout, &fluid_renderer_group_layout],
                    push_constant_ranges: &[],
                })),
                vertex_shader_relative_path: PathBuf::from("screenspace_fluid/particles.vert"),
                fragment_shader_relative_path: Some(PathBuf::from("screenspace_fluid/particles.frag")),
                rasterization_state: Some(rasterization_state::culling_none()),
                primitive_topology: wgpu::PrimitiveTopology::TriangleStrip,
                color_states: vec![
                    wgpu::ColorStateDescriptor {
                        format: Self::FORMAT_FLUID_DEPTH,
                        color_blend: wgpu::BlendDescriptor {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Min,
                        },
                        alpha_blend: wgpu::BlendDescriptor::REPLACE,
                        write_mask: wgpu::ColorWrite::ALL,
                    },
                    wgpu::ColorStateDescriptor {
                        format: Self::FORMAT_FLUID_THICKNESS,
                        color_blend: wgpu::BlendDescriptor {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha_blend: wgpu::BlendDescriptor::REPLACE,
                        write_mask: wgpu::ColorWrite::ALL,
                    },
                ],
                depth_stencil_state: Some(wgpu::DepthStencilStateDescriptor {
                    format: Screen::FORMAT_DEPTH,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil_front: wgpu::StencilStateFaceDescriptor::IGNORE,
                    stencil_back: wgpu::StencilStateFaceDescriptor::IGNORE,
                    stencil_read_mask: 0,
                    stencil_write_mask: 0,
                }),
                vertex_state: wgpu::VertexStateDescriptor {
                    index_format: wgpu::IndexFormat::Uint16,
                    vertex_buffers: &[],
                },
                sample_count: 1,
                sample_mask: !0,
                alpha_to_coverage_enabled: false,
            },
        );

        let pipeline_compose = pipeline_manager.create_compute_pipeline(
            device,
            shader_dir,
            ComputePipelineCreationDesc::new(
                Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    bind_group_layouts: &[&per_frame_bind_group_layout, &fluid_renderer_group_layout, &group_layout_compose.layout],
                    push_constant_ranges: &[],
                })),
                Path::new("screenspace_fluid/compose.comp"),
            ),
        );

        let screen_independent = ScreenIndependentProperties {
            group_layout_compose,
            pipeline_compose,
            pipeline_render_particles,
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
            depth: 1,
        };
        let texture_fluid_depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Texture: Fluid Depth"),
            size: target_textures_resolution,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::FORMAT_FLUID_DEPTH,
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT | wgpu::TextureUsage::SAMPLED,
        });
        let texture_view_fluid_depth = texture_fluid_depth.create_default_view();

        let texture_fluid_thickness = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Texture: Fluid Thickness"),
            size: target_textures_resolution,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::FORMAT_FLUID_THICKNESS,
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT | wgpu::TextureUsage::SAMPLED,
        });
        let texture_view_fluid_thickness = texture_fluid_thickness.create_default_view();

        let bind_group_compose = BindGroupBuilder::new(&screen_independent.group_layout_compose)
            .texture(&texture_view_fluid_depth)
            .texture(&texture_view_fluid_thickness)
            .texture(&backbuffer.texture_view())
            .create(device, "BindGroup: SSFluid, Final Compose");

        ScreenDependentProperties {
            texture_view_fluid_depth,
            texture_view_fluid_thickness,
            target_textures_resolution,
            bind_group_compose,
        }
    }

    pub fn on_window_resize(&mut self, device: &wgpu::Device, backbuffer: &HdrBackbuffer) {
        self.screen_dependent = Self::create_screen_dependent_properties(&self.screen_independent, device, backbuffer);
    }

    pub fn draw<'a>(
        &'a self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline_manager: &'a PipelineManager,
        depthbuffer: &wgpu::TextureView,
        per_frame_bind_group: &wgpu::BindGroup,
        fluid: &HybridFluid,
    ) {
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[
                    wgpu::RenderPassColorAttachmentDescriptor {
                        attachment: &self.screen_dependent.texture_view_fluid_depth,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                            store: true,
                        },
                    },
                    wgpu::RenderPassColorAttachmentDescriptor {
                        attachment: &self.screen_dependent.texture_view_fluid_thickness,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: true,
                        },
                    },
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                    attachment: depthbuffer,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: false,
                    }),
                    stencil_ops: None,
                }),
            });
            rpass.push_debug_group("screen space fluid, particles");

            rpass.set_bind_group(0, &per_frame_bind_group, &[]);
            rpass.set_bind_group(1, fluid.bind_group_renderer(), &[]);
            rpass.set_pipeline(pipeline_manager.get_render(&self.screen_independent.pipeline_render_particles));
            rpass.draw(0..4, 0..fluid.num_particles());

            rpass.pop_debug_group();
        }

        {
            const COMPUTE_LOCAL_SIZE_COMPOSE: wgpu::Extent3d = wgpu::Extent3d {
                width: 32,
                height: 32,
                depth: 1,
            };
            let compose_work_group = wgpu_utils::compute_group_size(self.screen_dependent.target_textures_resolution, COMPUTE_LOCAL_SIZE_COMPOSE);

            let mut cpass = encoder.begin_compute_pass();
            cpass.set_bind_group(0, &per_frame_bind_group, &[]);
            cpass.set_bind_group(1, fluid.bind_group_renderer(), &[]);
            cpass.set_bind_group(2, &self.screen_dependent.bind_group_compose, &[]);

            cpass.set_pipeline(pipeline_manager.get_compute(&self.screen_independent.pipeline_compose));
            cpass.dispatch(compose_work_group.width, compose_work_group.height, compose_work_group.depth);
        }
    }
}
