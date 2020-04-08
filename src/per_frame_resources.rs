use crate::camera;
use crate::wgpu_utils::bindings::*;

pub struct PerFrameResources {
    ubo_camera: camera::CameraUniformBuffer,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

impl PerFrameResources {
    pub fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = BindGroupLayoutBuilder::new()
            .next_binding_rendering(wgpu::BindingType::UniformBuffer { dynamic: false })
            .next_binding_all(wgpu::BindingType::Sampler { comparison: false })
            .create(device, "BindGroupLayout: PerFrameResources");

        let ubo_camera = camera::CameraUniformBuffer::new(&device);
        let trilinear_sampler = device.create_sampler(&simple_sampler(wgpu::AddressMode::ClampToEdge, wgpu::FilterMode::Linear));

        let bind_group = BindGroupBuilder::new(&bind_group_layout)
            .resource(ubo_camera.binding_resource())
            .sampler(&trilinear_sampler)
            .create(device, "BindGroup: PerFrameResources");

        PerFrameResources {
            ubo_camera,
            bind_group_layout: bind_group_layout.layout,
            bind_group,
        }
    }

    pub fn update_gpu_data(&self, encoder: &mut wgpu::CommandEncoder, device: &wgpu::Device, camera: &camera::Camera, aspect_ratio: f32) {
        self.ubo_camera.update_content(encoder, device, camera.fill_uniform_buffer(aspect_ratio));
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }
}
