use super::particle_renderer::ParticleRenderer;
use super::static_line_renderer::{LineVertex, StaticLineRenderer};
use super::volume_renderer::VolumeRenderer;
use crate::{
    hybrid_fluid::HybridFluid,
    scene::Scene,
    wgpu_utils::{pipelines::PipelineManager, shader::ShaderDirectory},
};

#[derive(Clone, Copy, Debug, EnumIter)]
pub enum FluidRenderingMode {
    None,
    Particles,
}

#[derive(Clone, Copy, Debug, EnumIter)]
pub enum VolumeVisualizationMode {
    None,
    Velocity,
    DivergenceUncorrected,
    PseudoPressure,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GlobalRenderSettingsUniformBufferContent {
    velocity_visualization_scale: f32,
    padding: cgmath::Point3<f32>,
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
    pub velocity_visualization_scale: f32,
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
            velocity_visualization_scale: 0.05,
        }
    }

    // Needs to be called whenever immutable scene properties change.
    pub fn on_new_scene(&mut self, device: &wgpu::Device, init_encoder: &mut wgpu::CommandEncoder, scene: &Scene) {
        let line_color = cgmath::vec3(0.0, 0.0, 0.0);
        let grid_extent = scene.fluid().grid_dimension();
        let min = scene.fluid_origin;
        let max = min + cgmath::vec3(grid_extent.width as f32, grid_extent.height as f32, grid_extent.depth as f32) * scene.fluid_to_world_scale;

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

    pub fn fill_global_uniform_buffer(&self) -> GlobalRenderSettingsUniformBufferContent {
        GlobalRenderSettingsUniformBufferContent {
            velocity_visualization_scale: self.velocity_visualization_scale,
            padding: cgmath::point3(0.0, 0.0, 0.0),
        }
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
                self.particle_renderer.draw(&mut rpass, pipeline_manager, &scene.fluid());
            }
        }
        match self.volume_visualization {
            VolumeVisualizationMode::None => {}
            VolumeVisualizationMode::Velocity => {
                self.volume_renderer.draw_volume_velocities(&mut rpass, pipeline_manager, &scene.fluid());
            }
            VolumeVisualizationMode::DivergenceUncorrected => {
                self.volume_renderer.draw_volume_divergence(&mut rpass, pipeline_manager, &scene.fluid());
            }
            VolumeVisualizationMode::PseudoPressure => {
                self.volume_renderer.draw_volume_pressure(&mut rpass, pipeline_manager, &scene.fluid());
            }
        }

        if self.enable_box_lines {
            self.bounds_line_renderer.draw(&mut rpass, pipeline_manager);
        }
    }
}
