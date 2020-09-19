use crate::camera;
use crate::timer;
use crate::wgpu_utils::binding_builder::*;
use crate::{renderer, wgpu_utils::*};
use uniformbuffer::UniformBuffer;

#[repr(C)]
#[derive(Clone, Copy)]
struct PerFrameUniformBufferContent {
    camera: camera::CameraUniformBufferContent,
    time: timer::FrameTimeUniformBufferContent,
    rendering: renderer::GlobalRenderSettingsUniformBufferContent,
}
unsafe impl bytemuck::Pod for PerFrameUniformBufferContent {}
unsafe impl bytemuck::Zeroable for PerFrameUniformBufferContent {}

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
        let trilinear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Sampler LinearClamp (global)"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let point_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Sampler NearestClamp (global)"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

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
        &mut self,
        queue: &wgpu::Queue,
        camera: camera::CameraUniformBufferContent,
        time: timer::FrameTimeUniformBufferContent,
        rendering: renderer::GlobalRenderSettingsUniformBufferContent,
    ) {
        self.ubo.update_content(queue, PerFrameUniformBufferContent { camera, time, rendering });
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }
}
