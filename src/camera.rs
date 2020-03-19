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

pub type CameraUniformBuffer = super::uniformbuffer::UniformBuffer<CameraUniformBufferContent>;
