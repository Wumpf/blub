use crate::camera;
use crate::renderer::scene_renderer;
use crate::timer;
use crate::wgpu_utils::binding_builder::*;
use crate::wgpu_utils::*;
use uniformbuffer::UniformBuffer;

#[repr(C)]
#[derive(Clone, Copy)]
struct PerFrameUniformBufferContent {
    camera: camera::CameraUniformBufferContent,
    time: timer::FrameTimeUniformBufferContent,
    rendering: scene_renderer::GlobalRenderSettingsUniformBufferContent,
}

type PerFrameUniformBuffer = UniformBuffer<PerFrameUniformBufferContent>;

pub struct PerFrameResources {
    ubo: PerFrameUniformBuffer,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

impl PerFrameResources {
    pub fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = BindGroupLayoutBuilder::new()
            .next_binding_all(binding_glsl::uniform())
            .next_binding_all(binding_glsl::sampler())
            .next_binding_all(binding_glsl::sampler())
            .create(device, "BindGroupLayout: PerFrameResources");

        let ubo = PerFrameUniformBuffer::new(&device);
        let trilinear_sampler = device.create_sampler(&simple_sampler(wgpu::AddressMode::ClampToEdge, wgpu::FilterMode::Linear));
        let point_sampler = device.create_sampler(&simple_sampler(wgpu::AddressMode::ClampToEdge, wgpu::FilterMode::Nearest));

        let bind_group = BindGroupBuilder::new(&bind_group_layout)
            .resource(ubo.binding_resource())
            .sampler(&trilinear_sampler)
            .sampler(&point_sampler)
            .create(device, "BindGroup: PerFrameResources");

        PerFrameResources {
            ubo,
            bind_group_layout: bind_group_layout.layout,
            bind_group,
        }
    }

    pub fn update_gpu_data(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        camera: camera::CameraUniformBufferContent,
        time: timer::FrameTimeUniformBufferContent,
        rendering: scene_renderer::GlobalRenderSettingsUniformBufferContent,
    ) {
        self.ubo
            .update_content(encoder, device, PerFrameUniformBufferContent { camera, time, rendering });
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }
}
