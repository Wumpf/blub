use super::rendertimer::RenderTimer;
use super::wgpu_utils::uniformbuffer::*;
use cgmath::prelude::*;
use enumflags2::BitFlags;
use winit::event::{DeviceEvent, ElementState, KeyboardInput, VirtualKeyCode, WindowEvent};

#[cfg_attr(rustfmt, rustfmt_skip)]
const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

#[derive(BitFlags, Copy, Clone, Debug, PartialEq)]
enum MoveCommands {
    Left = 0b0001,
    Right = 0b0010,
    Forwards = 0b0100,
    Backwards = 0b1000,
    SpeedUp = 0b1_0000,
}

pub struct Camera {
    pub position: cgmath::Point3<f32>,
    pub direction: cgmath::Vector3<f32>,
    rotational_up: cgmath::Vector3<f32>,

    movement_locked: bool,
    active_move_commands: BitFlags<MoveCommands>,
    mouse_delta: (f64, f64),

    translation_speed: f32,
    rotation_speed: f32,
}

impl Camera {
    pub fn new() -> Camera {
        let position = cgmath::Point3::new(-100.0f32, 100.0, 100.0);
        Camera {
            position,
            direction: cgmath::Point3::new(0f32, 0.0, 0.0) - position,
            rotational_up: cgmath::Vector3::unit_y(),

            movement_locked: true,
            active_move_commands: Default::default(),
            mouse_delta: (0.0, 0.0),

            translation_speed: 8.0,
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
                    VirtualKeyCode::S | VirtualKeyCode::Down => BitFlags::from(MoveCommands::Backwards),
                    VirtualKeyCode::A | VirtualKeyCode::Left => BitFlags::from(MoveCommands::Left),
                    VirtualKeyCode::D | VirtualKeyCode::Right => BitFlags::from(MoveCommands::Right),
                    VirtualKeyCode::W | VirtualKeyCode::Up => BitFlags::from(MoveCommands::Forwards),
                    VirtualKeyCode::LShift => BitFlags::from(MoveCommands::SpeedUp),
                    _ => Default::default(),
                };
                match state {
                    ElementState::Pressed => self.active_move_commands.insert(direction),
                    ElementState::Released => self.active_move_commands.remove(direction),
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
            let right = self.direction.cross(self.rotational_up).normalize();

            let mut translation = (self.active_move_commands.contains(MoveCommands::Forwards) as i32 as f32
                - self.active_move_commands.contains(MoveCommands::Backwards) as i32 as f32)
                * self.direction;
            translation += (self.active_move_commands.contains(MoveCommands::Right) as i32 as f32
                - self.active_move_commands.contains(MoveCommands::Left) as i32 as f32)
                * right;
            translation *= timer.frame_delta_time().as_secs_f32() * self.translation_speed;
            if self.active_move_commands.contains(MoveCommands::SpeedUp) {
                translation *= 4.0;
            }

            let rotation_updown = cgmath::Quaternion::from_axis_angle(right, cgmath::Rad(-self.mouse_delta.1 as f32 * self.rotation_speed));
            let rotation_leftright =
                cgmath::Quaternion::from_axis_angle(self.rotational_up, cgmath::Rad(-self.mouse_delta.0 as f32 * self.rotation_speed));
            self.direction = (rotation_updown + rotation_leftright).rotate_vector(self.direction).normalize();

            self.position += translation;
        }

        self.mouse_delta = (0.0, 0.0);
    }

    fn view_projection(&self, aspect_ratio: f32) -> cgmath::Matrix4<f32> {
        let projection = cgmath::perspective(cgmath::Deg(80f32), aspect_ratio, 0.1, 1000.0);
        let view = cgmath::Matrix4::look_at_dir(self.position, self.direction, self.rotational_up);
        OPENGL_TO_WGPU_MATRIX * projection * view
    }

    pub fn fill_uniform_buffer(&self, aspect_ratio: f32) -> CameraUniformBufferContent {
        let right = self.direction.cross(self.rotational_up).normalize();
        let up = right.cross(self.direction).normalize();

        CameraUniformBufferContent {
            view_projection: self.view_projection(aspect_ratio),
            position: self.position.into(),
            right: right.into(),
            up: up.into(),
            direction: self.direction.into(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CameraUniformBufferContent {
    pub view_projection: cgmath::Matrix4<f32>,
    pub position: PaddedPoint3,
    pub right: PaddedVector3,
    pub up: PaddedVector3,
    pub direction: PaddedVector3,
}

pub type CameraUniformBuffer = UniformBuffer<CameraUniformBufferContent>;
