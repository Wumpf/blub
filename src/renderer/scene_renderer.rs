use super::{
    background::Background,
    mesh_renderer::MeshRenderer,
    particle_renderer::ParticleRenderer,
    screenspace_fluid::ScreenSpaceFluid,
    static_line_renderer::{LineVertex, StaticLineRenderer},
    volume_renderer::{VolumeRenderer, VolumeVisualizationMode},
};
use crate::{
    render_output::hdr_backbuffer::HdrBackbuffer,
    scene::Scene,
    simulation::HybridFluid,
    wgpu_utils::{pipelines::PipelineManager, shader::ShaderDirectory},
};

use cgmath::EuclideanSpace;
use std::path::Path;
#[derive(Clone, Copy, Debug, EnumIter)]
pub enum FluidRenderingMode {
    None,
    ScreenSpaceFluid,
    Particles,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GlobalRenderSettingsUniformBufferContent {
    fluid_origin: cgmath::Point3<f32>,
    fluid_grid_to_world_scale: f32,
    velocity_visualization_scale: f32,
    fluid_particle_radius: f32,
    padding: cgmath::Point2<f32>,
}

// What renders the scene (so everything except ui!)
// Maintains both configuration and necessary data structures, but doesn't shut down when a scene is swapped out.
pub struct SceneRenderer {
    particle_renderer: ParticleRenderer,
    screenspace_fluid: ScreenSpaceFluid,
    volume_renderer: VolumeRenderer,
    bounds_line_renderer: StaticLineRenderer,
    mesh_renderer: MeshRenderer,
    background_and_lighting: Background,

    pub fluid_rendering_mode: FluidRenderingMode,
    pub volume_visualization: VolumeVisualizationMode,
    pub particle_radius_factor: f32,
    pub enable_box_lines: bool,
    pub velocity_visualization_scale: f32,
}

impl SceneRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        shader_dir: &ShaderDirectory,
        pipeline_manager: &mut PipelineManager,
        per_frame_bind_group_layout: &wgpu::BindGroupLayout,
        backbuffer: &HdrBackbuffer,
    ) -> Self {
        let fluid_renderer_group_layout = &HybridFluid::get_or_create_group_layout_renderer(device).layout;

        let background_and_lighting = Background::new(
            Path::new("background"),
            device,
            queue,
            shader_dir,
            pipeline_manager,
            per_frame_bind_group_layout,
        )
        .unwrap();

        SceneRenderer {
            screenspace_fluid: ScreenSpaceFluid::new(
                device,
                shader_dir,
                pipeline_manager,
                per_frame_bind_group_layout,
                fluid_renderer_group_layout,
                background_and_lighting.bind_group_layout(),
                backbuffer,
            ),
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
            mesh_renderer: MeshRenderer::new(
                device,
                shader_dir,
                pipeline_manager,
                per_frame_bind_group_layout,
                background_and_lighting.bind_group_layout(),
            ),
            background_and_lighting,

            fluid_rendering_mode: FluidRenderingMode::ScreenSpaceFluid,
            volume_visualization: VolumeVisualizationMode::None,
            particle_radius_factor: 0.7,
            enable_box_lines: true,
            velocity_visualization_scale: 0.008,
        }
    }

    // Needs to be called whenever immutable scene properties change.
    pub fn on_new_scene(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, scene: &Scene) {
        self.mesh_renderer.on_new_scene(device, &scene.models);

        let line_color = cgmath::vec3(0.0, 0.0, 0.0);
        let grid_extent = scene.config().fluid.grid_dimension;
        let min = scene.config().fluid.world_position;
        let max = min + grid_extent.cast().unwrap().to_vec() * scene.config().fluid.grid_to_world_scale;

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
            queue,
        );
    }

    pub fn fill_global_uniform_buffer(&self, scene: &Scene) -> GlobalRenderSettingsUniformBufferContent {
        let fluid_particle_radius =
            scene.config().fluid.grid_to_world_scale / (HybridFluid::PARTICLES_PER_GRID_CELL as f32).powf(1.0 / 3.0) * self.particle_radius_factor;

        GlobalRenderSettingsUniformBufferContent {
            fluid_origin: scene.config().fluid.world_position,
            fluid_grid_to_world_scale: scene.config().fluid.grid_to_world_scale,
            velocity_visualization_scale: self.velocity_visualization_scale,
            fluid_particle_radius,
            padding: cgmath::point2(0.0, 0.0),
        }
    }

    pub fn on_window_resize(&mut self, device: &wgpu::Device, backbuffer: &HdrBackbuffer) {
        self.screenspace_fluid.on_window_resize(device, backbuffer);
    }

    pub fn draw(
        &self,
        scene: &Scene,
        encoder: &mut wgpu::CommandEncoder,
        pipeline_manager: &PipelineManager,
        backbuffer: &wgpu::TextureView,
        depthbuffer: &wgpu::TextureView,
        per_frame_bind_group: &wgpu::BindGroup,
    ) {
        wgpu_scope!(encoder, "SceneRenderer.draw");
        {
            // Opaque
            wgpu_scope!(encoder, "opaque", || {
                let mut rpass_backbuffer = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                        attachment: backbuffer,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: true,
                        },
                    }],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                        attachment: depthbuffer,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: true,
                        }),
                        stencil_ops: None,
                    }),
                });
                rpass_backbuffer.set_bind_group(0, per_frame_bind_group, &[]);

                match self.fluid_rendering_mode {
                    FluidRenderingMode::None => {}
                    FluidRenderingMode::ScreenSpaceFluid => {
                        // Handled earlier!
                    }
                    FluidRenderingMode::Particles => {
                        self.particle_renderer.draw(&mut rpass_backbuffer, pipeline_manager, &scene.fluid());
                    }
                }

                self.mesh_renderer.draw(
                    &mut rpass_backbuffer,
                    pipeline_manager,
                    self.background_and_lighting.bind_group(),
                    &scene.models,
                );
                self.volume_renderer
                    .draw(&mut rpass_backbuffer, pipeline_manager, &scene.fluid(), self.volume_visualization);

                if self.enable_box_lines {
                    self.bounds_line_renderer.draw(&mut rpass_backbuffer, pipeline_manager);
                }

                // Background.. not really opaque but we re-use the same rpass.
                self.background_and_lighting.draw(&mut rpass_backbuffer, pipeline_manager);
            });

            // Transparent
            wgpu_scope!(encoder, "transparent", || {
                if let FluidRenderingMode::ScreenSpaceFluid = self.fluid_rendering_mode {
                    self.screenspace_fluid.draw(
                        &mut encoder,
                        pipeline_manager,
                        depthbuffer,
                        per_frame_bind_group,
                        self.background_and_lighting.bind_group(),
                        &scene.fluid(),
                    );
                }
            });
        }
    }
}
