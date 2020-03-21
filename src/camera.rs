use super::rendertimer::RenderTimer;
use cgmath::prelude::*;
use enumflags2::BitFlags;
use winit::event::{DeviceEvent, ElementState, KeyboardInput, VirtualKeyCode, WindowEvent};

#[cfg_attr(rustfmt, rustfmt_skip)]
const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, -1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

#[derive(BitFlags, Copy, Clone, Debug, PartialEq)]
enum MoveDirection {
    Left = 0b0001,
    Right = 0b0010,
    Forwards = 0b0100,
    Backwards = 0b1000,
}

pub struct Camera {
    pub position: cgmath::Point3<f32>,
    pub direction: cgmath::Vector3<f32>,
    up: cgmath::Vector3<f32>,

    movement_locked: bool,
    move_directions: BitFlags<MoveDirection>,
    mouse_delta: (f64, f64),

    translation_speed: f32,
    rotation_speed: f32,
}

impl Camera {
    pub fn new() -> Camera {
        let position = cgmath::Point3::new(1.5f32, -5.0, 3.0);
        Camera {
            position,
            direction: cgmath::Point3::new(0f32, 0.0, 0.0) - position,
            up: cgmath::Vector3::unit_y(),

            movement_locked: true,
            move_directions: Default::default(),
            mouse_delta: (0.0, 0.0),

            translation_speed: 1.0,
            rotation_speed: 0.001,
        }
    }

    pub fn on_window_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::KeyboardInput {
                // Ugh, that should probably be ScanCode otherwise non-querty/quertz people are going to be angry with me.
                // But using ScanCode is cumbersome since there's no defines. So whatever...
                input:
                    KeyboardInput {
                        virtual_keycode: Some(virtual_keycode),
                        state,
                        ..
                    },
                ..
            } => {
                let direction = match virtual_keycode {
                    VirtualKeyCode::S | VirtualKeyCode::Down => BitFlags::from(MoveDirection::Backwards),
                    VirtualKeyCode::A | VirtualKeyCode::Left => BitFlags::from(MoveDirection::Left),
                    VirtualKeyCode::D | VirtualKeyCode::Right => BitFlags::from(MoveDirection::Right),
                    VirtualKeyCode::W | VirtualKeyCode::Up => BitFlags::from(MoveDirection::Forwards),
                    _ => Default::default(),
                };
                match state {
                    ElementState::Pressed => self.move_directions.insert(direction),

                    ElementState::Released => self.move_directions.remove(direction),
                };
            }
            WindowEvent::MouseInput { button, state, .. } => {
                if *button == winit::event::MouseButton::Right {
                    self.movement_locked = *state == ElementState::Released;
                }
            }
            _ => {}
        }
    }

    pub fn on_device_event(&mut self, event: &DeviceEvent) {
        match event {
            DeviceEvent::MouseMotion { delta } => {
                self.mouse_delta = (self.mouse_delta.0 + delta.0, self.mouse_delta.1 + delta.1);
            }
            _ => {}
        }
    }

    pub fn update(&mut self, timer: &RenderTimer) {
        if self.movement_locked == false {
            let right = self.direction.cross(self.up).normalize();

            let mut translation = (self.move_directions.contains(MoveDirection::Forwards) as i32 as f32
                - self.move_directions.contains(MoveDirection::Backwards) as i32 as f32)
                * self.direction;
            translation += (self.move_directions.contains(MoveDirection::Right) as i32 as f32
                - self.move_directions.contains(MoveDirection::Left) as i32 as f32)
                * right;
            translation *= timer.frame_delta_time().as_secs_f32() * self.translation_speed;

            let rotation_updown = cgmath::Quaternion::from_axis_angle(right, cgmath::Rad(-self.mouse_delta.1 as f32 * self.rotation_speed));
            let rotation_leftright = cgmath::Quaternion::from_axis_angle(self.up, cgmath::Rad(-self.mouse_delta.0 as f32 * self.rotation_speed));
            self.direction = (rotation_updown + rotation_leftright).rotate_vector(self.direction);

            self.position += translation;
        }

        self.mouse_delta = (0.0, 0.0);
    }

    pub fn view_projection(&self, aspect_ratio: f32) -> cgmath::Matrix4<f32> {
        let projection = cgmath::perspective(cgmath::Deg(45f32), aspect_ratio, 1.0, 1000.0);
        let view = cgmath::Matrix4::look_at_dir(self.position, self.direction, self.up);
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
