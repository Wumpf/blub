//use cgmath::prelude::*;

#[cfg_attr(rustfmt, rustfmt_skip)]
const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, -1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);
pub struct Camera;

impl Camera {
    pub fn view_projection(aspect_ratio: f32) -> cgmath::Matrix4<f32> {
        let projection = cgmath::perspective(cgmath::Deg(45f32), aspect_ratio, 1.0, 10.0);
        let view = cgmath::Matrix4::look_at(
            cgmath::Point3::new(1.5f32, -5.0, 3.0),
            cgmath::Point3::new(0f32, 0.0, 0.0),
            cgmath::Vector3::unit_z(),
        );
        OPENGL_TO_WGPU_MATRIX * projection * view
    }
}

// todo: (semi-)generate this from the shader code?
#[repr(C)]
pub struct CameraUniformBufferContent {
    view_projection: cgmath::Matrix4<f32>,
}

pub struct CameraUniformBuffer {
    //cpu_repr: CameraUniformBufferContent,
    pub buffer: wgpu::Buffer,
}

impl CameraUniformBuffer {
    pub fn new(device: &wgpu::Device) -> CameraUniformBuffer {
        let mapped_buffer = device.create_buffer_mapped(1, wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST);
        let buffer = mapped_buffer.fill_from_slice(&[Camera::view_projection(1.0)]);

        CameraUniformBuffer {
            // cpu_repr: CameraUniformBufferContent {
            //     view_projection: cgmath::Matrix4::identity(),
            // },
            buffer,
        }
    }
}
