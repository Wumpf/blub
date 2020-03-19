use cgmath::prelude::*;

#[cfg_attr(rustfmt, rustfmt_skip)]
const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, -1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);
pub struct Camera {
    pub position: cgmath::Point3<f32>,
    pub direction: cgmath::Vector3<f32>,
}

impl Camera {
    pub fn new() -> Camera {
        let position = cgmath::Point3::new(1.5f32, -5.0, 3.0);
        Camera {
            position,
            direction: cgmath::Point3::new(0f32, 0.0, 0.0) - position,
        }
    }

    pub fn update(&mut self, time_startup: std::time::Duration) {
        self.position = cgmath::Point3::new(time_startup.as_secs_f32().cos(), 0.0, time_startup.as_secs_f32().sin()) * 4.0;
        self.direction = cgmath::Point3::origin() - self.position;
    }

    pub fn view_projection(&self, aspect_ratio: f32) -> cgmath::Matrix4<f32> {
        let projection = cgmath::perspective(cgmath::Deg(45f32), aspect_ratio, 1.0, 10.0);
        let view = cgmath::Matrix4::look_at_dir(self.position, self.direction, cgmath::Vector3::unit_y());
        OPENGL_TO_WGPU_MATRIX * projection * view
    }
}

// todo: (semi-)generate this from the shader code?
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CameraUniformBufferContent {
    pub view_projection: cgmath::Matrix4<f32>,
}

pub struct CameraUniformBuffer {
    buffer: wgpu::Buffer,
}

impl CameraUniformBuffer {
    pub fn new(device: &wgpu::Device) -> CameraUniformBuffer {
        let buffer = device
            .create_buffer_mapped(1, wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST)
            .fill_from_slice(&[cgmath::Matrix4::<f32>::identity()]);

        CameraUniformBuffer { buffer }
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub fn update_content(&self, encoder: &mut wgpu::CommandEncoder, device: &wgpu::Device, content: CameraUniformBufferContent) {
        let buffer = device.create_buffer_mapped(1, wgpu::BufferUsage::COPY_SRC).fill_from_slice(&[content]);
        encoder.copy_buffer_to_buffer(&buffer, 0, &self.buffer, 0, std::mem::size_of::<CameraUniformBufferContent>() as u64);
    }
}
