use crate::hybrid_fluid::HybridFluid;
use crate::particle_renderer::ParticleRenderer;
use crate::wgpu_utils::shader::ShaderDirectory;

// The simulated scene.
pub struct Scene {
    hybrid_fluid: HybridFluid,
}

impl Scene {
    pub fn new(
        device: &wgpu::Device,
        command_queue: &wgpu::Queue,
        shader_dir: &ShaderDirectory,
        per_frame_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let mut hybrid_fluid = HybridFluid::new(
            device,
            wgpu::Extent3d {
                width: 128,
                height: 64,
                depth: 64,
            },
            2000000,
            shader_dir,
            per_frame_bind_group_layout,
        );

        let mut init_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Scene Init Encoder"),
        });

        hybrid_fluid.add_fluid_cube(
            device,
            &mut init_encoder,
            cgmath::Point3::new(0.0, 0.0, 0.0),
            cgmath::Point3::new(63.0, 40.0, 63.0),
        );

        // Should this be bundled with others?
        command_queue.submit(&[init_encoder.finish()]);
        device.poll(wgpu::Maintain::Wait);

        Scene { hybrid_fluid }
    }

    pub fn try_reload_shaders(&mut self, device: &wgpu::Device, shader_dir: &ShaderDirectory) {
        self.hybrid_fluid.try_reload_shaders(device, shader_dir);
    }

    pub fn step<'a>(&'a self, cpass: &mut wgpu::ComputePass<'a>) {
        self.hybrid_fluid.step(cpass);
    }
}

// What renders the scene
// (so everything except ui!)
pub struct SceneRenderer {
    pub particle_renderer: ParticleRenderer,
}

impl SceneRenderer {
    pub fn new(device: &wgpu::Device, shader_dir: &ShaderDirectory, per_frame_bind_group_layout: &wgpu::BindGroupLayout) -> Self {
        // Creates a new group layout for rendering for decoupling Scene from SceneRenderer
        SceneRenderer {
            particle_renderer: ParticleRenderer::new(
                &device,
                &shader_dir,
                per_frame_bind_group_layout,
                &HybridFluid::get_or_create_group_layout_renderer(device).layout,
            ),
        }
    }

    pub fn draw(
        &self,
        scene: &Scene,
        encoder: &mut wgpu::CommandEncoder,
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
        self.particle_renderer.draw(&mut rpass, &scene.hybrid_fluid);
    }

    pub fn try_reload_shaders(&mut self, device: &wgpu::Device, shader_dir: &ShaderDirectory) {
        self.particle_renderer.try_reload_shaders(device, shader_dir);
    }
}
