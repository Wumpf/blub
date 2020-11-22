use crate::timer;
use crate::{camera, render_output::screen};
use crate::{renderer, wgpu_utils::*};
use uniformbuffer::UniformBuffer;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GlobalUBOContent {
    camera: camera::CameraUniformBufferContent,
    time: timer::FrameTimeUniformBufferContent,
    rendering: renderer::GlobalRenderSettingsUniformBufferContent,
    screen: screen::ScreenUniformBufferContent,
}
unsafe impl bytemuck::Pod for GlobalUBOContent {}
unsafe impl bytemuck::Zeroable for GlobalUBOContent {}

pub type GlobalUBO = UniformBuffer<GlobalUBOContent>;

pub fn update_global_ubo(
    ubo: &mut GlobalUBO,
    queue: &wgpu::Queue,
    camera: camera::CameraUniformBufferContent,
    time: timer::FrameTimeUniformBufferContent,
    rendering: renderer::GlobalRenderSettingsUniformBufferContent,
    screen: screen::ScreenUniformBufferContent,
) {
    ubo.update_content(
        queue,
        GlobalUBOContent {
            camera,
            time,
            rendering,
            screen,
        },
    );
}
