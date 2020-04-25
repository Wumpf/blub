use crate::hybrid_fluid::HybridFluid;
use crate::wgpu_utils::shader::ShaderDirectory;

// The simulated scene.
pub struct Scene {
    pub hybrid_fluid: HybridFluid,
}

impl Scene {
    pub fn new(device: &wgpu::Device, shader_dir: &ShaderDirectory, per_frame_resource_group_layout: &wgpu::BindGroupLayout) -> Self {
        Scene {
            hybrid_fluid: HybridFluid::new(
                device,
                wgpu::Extent3d {
                    width: 128,
                    height: 64,
                    depth: 64,
                },
                2000000,
                shader_dir,
                per_frame_resource_group_layout,
            ),
        }
    }

    pub fn reset(&mut self, device: &wgpu::Device, command_queue: &wgpu::Queue) {
        let mut init_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Particle Init Encoder"),
        });

        self.hybrid_fluid.reset();
        self.hybrid_fluid.add_fluid_cube(
            device,
            &mut init_encoder,
            cgmath::Point3::new(0.0, 0.0, 0.0),
            cgmath::Point3::new(64.0, 40.0, 64.0),
        );

        command_queue.submit(&[init_encoder.finish()]);
        device.poll(wgpu::Maintain::Wait);
    }

    pub fn try_reload_shaders(&mut self, device: &wgpu::Device, shader_dir: &ShaderDirectory) {
        self.hybrid_fluid.try_reload_shaders(device, shader_dir);
    }

    pub fn step<'a>(&'a self, cpass: &mut wgpu::ComputePass<'a>) {
        self.hybrid_fluid.step(cpass);
    }
}
