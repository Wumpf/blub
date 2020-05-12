use crate::hybrid_fluid::HybridFluid;
use crate::particle_renderer::ParticleRenderer;
use crate::static_line_renderer::{LineVertex, StaticLineRenderer};
use crate::volume_renderer::VolumeRenderer;
use crate::wgpu_utils::{pipelines::PipelineManager, shader::ShaderDirectory};

// Scene data & simulation.
pub struct Scene {
    hybrid_fluid: HybridFluid,
    fluid_domain_min: cgmath::Point3<f32>,
    fluid_domain_max: cgmath::Point3<f32>,
}

impl Scene {
    pub fn new(
        device: &wgpu::Device,
        init_encoder: &mut wgpu::CommandEncoder,
        shader_dir: &ShaderDirectory,
        pipeline_manager: &mut PipelineManager,
        per_frame_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let fluid_domain_min = cgmath::Point3::new(0.0, 0.0, 0.0);
        let fluid_domain_max = cgmath::Point3::new(128.0, 64.0, 64.0);

        let grid_dimension = wgpu::Extent3d {
            width: (fluid_domain_max.x - fluid_domain_min.x) as u32,
            height: (fluid_domain_max.y - fluid_domain_min.y) as u32,
            depth: (fluid_domain_max.z - fluid_domain_min.z) as u32,
        };

        let mut hybrid_fluid = HybridFluid::new(device, grid_dimension, 2000000, shader_dir, pipeline_manager, per_frame_bind_group_layout);

        hybrid_fluid.add_fluid_cube(
            device,
            init_encoder,
            cgmath::Point3::new(0.0, 0.0, 0.0),
            cgmath::Point3::new(63.0, 40.0, 63.0),
        );

        Scene {
            hybrid_fluid,
            fluid_domain_min,
            fluid_domain_max,
        }
    }

    pub fn step<'a>(&'a self, cpass: &mut wgpu::ComputePass<'a>, pipeline_manager: &'a PipelineManager) {
        self.hybrid_fluid.step(cpass, pipeline_manager);
    }
}

#[derive(Clone, Copy, Debug, EnumIter)]
pub enum FluidRenderingMode {
    None,
    Particles,
}

#[derive(Clone, Copy, Debug, EnumIter)]
pub enum VolumeVisualizationMode {
    None,
    Velocity,
}

// What renders the scene (so everything except ui!)
// Maintains both configuration and necessary data structures, but doesn't shut down when a scene is swapped out.
pub struct SceneRenderer {
    particle_renderer: ParticleRenderer,
    volume_renderer: VolumeRenderer,
    bounds_line_renderer: StaticLineRenderer,

    pub fluid_rendering_mode: FluidRenderingMode,
    pub volume_visualization: VolumeVisualizationMode,
    pub enable_box_lines: bool,
}

impl SceneRenderer {
    pub fn new(
        device: &wgpu::Device,
        shader_dir: &ShaderDirectory,
        pipeline_manager: &mut PipelineManager,
        per_frame_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let fluid_renderer_group_layout = &HybridFluid::get_or_create_group_layout_renderer(device).layout;
        SceneRenderer {
            particle_renderer: ParticleRenderer::new(
                device,
                shader_dir,
                pipeline_manager,
                per_frame_bind_group_layout,
                fluid_renderer_group_layout,
            ),
            volume_renderer: VolumeRenderer::new(
                device,
                shader_dir,
                pipeline_manager,
                per_frame_bind_group_layout,
                fluid_renderer_group_layout,
            ),
            bounds_line_renderer: StaticLineRenderer::new(device, shader_dir, pipeline_manager, per_frame_bind_group_layout, 128),

            fluid_rendering_mode: FluidRenderingMode::Particles,
            volume_visualization: VolumeVisualizationMode::None,
            enable_box_lines: true,
        }
    }

    // Needs to be called whenever immutable scene properties change.
    pub fn on_new_scene(&mut self, device: &wgpu::Device, init_encoder: &mut wgpu::CommandEncoder, scene: &Scene) {
        let line_color = cgmath::Vector3::new(0.0, 0.0, 0.0);
        let min = scene.fluid_domain_min;
        let max = scene.fluid_domain_max;

        self.bounds_line_renderer.clear_lines();
        self.bounds_line_renderer.add_lines(
            &[
                // left
                LineVertex::new(cgmath::point3(min.x, min.y, max.z), line_color),
                LineVertex::new(cgmath::point3(max.x, min.y, max.z), line_color),
                LineVertex::new(cgmath::point3(max.x, min.y, max.z), line_color),
                LineVertex::new(cgmath::point3(max.x, max.y, max.z), line_color),
                LineVertex::new(cgmath::point3(max.x, max.y, max.z), line_color),
                LineVertex::new(cgmath::point3(min.x, max.y, max.z), line_color),
                LineVertex::new(cgmath::point3(min.x, max.y, max.z), line_color),
                LineVertex::new(cgmath::point3(min.x, min.y, max.z), line_color),
                // right
                LineVertex::new(cgmath::point3(min.x, min.y, min.z), line_color),
                LineVertex::new(cgmath::point3(max.x, min.y, min.z), line_color),
                LineVertex::new(cgmath::point3(max.x, min.y, min.z), line_color),
                LineVertex::new(cgmath::point3(max.x, max.y, min.z), line_color),
                LineVertex::new(cgmath::point3(max.x, max.y, min.z), line_color),
                LineVertex::new(cgmath::point3(min.x, max.y, min.z), line_color),
                LineVertex::new(cgmath::point3(min.x, max.y, min.z), line_color),
                LineVertex::new(cgmath::point3(min.x, min.y, min.z), line_color),
                // between
                LineVertex::new(cgmath::point3(min.x, min.y, min.z), line_color),
                LineVertex::new(cgmath::point3(min.x, min.y, max.z), line_color),
                LineVertex::new(cgmath::point3(max.x, min.y, min.z), line_color),
                LineVertex::new(cgmath::point3(max.x, min.y, max.z), line_color),
                LineVertex::new(cgmath::point3(max.x, max.y, min.z), line_color),
                LineVertex::new(cgmath::point3(max.x, max.y, max.z), line_color),
                LineVertex::new(cgmath::point3(min.x, max.y, min.z), line_color),
                LineVertex::new(cgmath::point3(min.x, max.y, max.z), line_color),
            ],
            device,
            init_encoder,
        );
    }

    pub fn draw(
        &self,
        scene: &Scene,
        encoder: &mut wgpu::CommandEncoder,
        pipeline_manager: &PipelineManager,
        backbuffer: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        per_frame_bind_group: &wgpu::BindGroup,
    ) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                attachment: backbuffer,
                resolve_target: None,
                load_op: wgpu::LoadOp::Clear,
                store_op: wgpu::StoreOp::Store,
                clear_color: wgpu::Color {
                    r: 0.1,
                    g: 0.2,
                    b: 0.3,
                    a: 1.0,
                },
            }],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                attachment: depth,
                depth_load_op: wgpu::LoadOp::Clear,
                depth_store_op: wgpu::StoreOp::Store,
                stencil_load_op: wgpu::LoadOp::Clear,
                stencil_store_op: wgpu::StoreOp::Store,
                clear_depth: 1.0,
                clear_stencil: 0,
            }),
        });

        rpass.set_bind_group(0, per_frame_bind_group, &[]);

        match self.fluid_rendering_mode {
            FluidRenderingMode::None => {}
            FluidRenderingMode::Particles => {
                self.particle_renderer.draw(&mut rpass, pipeline_manager, &scene.hybrid_fluid);
            }
        }
        match self.volume_visualization {
            VolumeVisualizationMode::None => {}
            VolumeVisualizationMode::Velocity => {
                self.volume_renderer
                    .draw_volume_velocities(&mut rpass, pipeline_manager, &scene.hybrid_fluid);
            }
        }

        if self.enable_box_lines {
            self.bounds_line_renderer.draw(&mut rpass, pipeline_manager);
        }
    }
}
