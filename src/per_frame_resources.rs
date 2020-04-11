use crate::camera;
use crate::rendertimer::RenderTimer;
use crate::wgpu_utils::binding_builder::*;
use crate::wgpu_utils::*;
use uniformbuffer::UniformBuffer;

#[repr(C)]
#[derive(Clone, Copy)]
struct PerFrameUniformBufferContent {
    pub camera: camera::CameraUniformBufferContent,

    pub total_passed_time: f32,
    pub delta_time: f32,
    pub padding0: f32,
    pub padding1: f32,
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
            .next_binding_rendering(binding_glsl::uniform())
            .next_binding_all(binding_glsl::sampler())
            .create(device, "BindGroupLayout: PerFrameResources");

        let ubo = PerFrameUniformBuffer::new(&device);
        let trilinear_sampler = device.create_sampler(&simple_sampler(wgpu::AddressMode::ClampToEdge, wgpu::FilterMode::Linear));

        let bind_group = BindGroupBuilder::new(&bind_group_layout)
            .resource(ubo.binding_resource())
            .sampler(&trilinear_sampler)
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
        camera: &camera::Camera,
        timer: &RenderTimer,
        aspect_ratio: f32,
    ) {
        self.ubo.update_content(
            encoder,
            device,
            PerFrameUniformBufferContent {
                camera: camera.fill_uniform_buffer(aspect_ratio),

                total_passed_time: timer.time_since_start().as_secs_f32(),
                delta_time: timer.frame_delta_time().as_secs_f32().min(0.0000000001),
                padding0: 0.0,
                padding1: 0.0,
            },
        );
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }
}
