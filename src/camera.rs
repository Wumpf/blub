use super::timer::Timer;
use super::wgpu_utils::uniformbuffer::*;
use cgmath::prelude::*;
use enumflags2::{bitflags, BitFlags};
use winit::event::{DeviceEvent, ElementState, KeyboardInput, VirtualKeyCode, WindowEvent};

#[cfg_attr(rustfmt, rustfmt_skip)]
const OPENGL_PROJECTION_TO_WGPU_PROJECTION: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

const VERTICAL_FOV: cgmath::Deg<f32> = cgmath::Deg(80f32);

#[bitflags]
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq)]
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
        let position = cgmath::Point3::new(1.0f32, 1.0, 1.0);
        Camera {
            position,
            direction: (cgmath::Point3::new(0f32, 0.0, 0.0) - position).normalize(),
            rotational_up: cgmath::Vector3::unit_y(),

            movement_locked: true,
            active_move_commands: Default::default(),
            mouse_delta: (0.0, 0.0),

            translation_speed: 0.5,
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

    pub fn update(&mut self, timer: &Timer) {
        if self.movement_locked == false {
            let right = self.direction.cross(self.rotational_up).normalize();

            let mut translation = (self.active_move_commands.contains(MoveCommands::Forwards) as i32 as f32
                - self.active_move_commands.contains(MoveCommands::Backwards) as i32 as f32)
                * self.direction;
            translation += (self.active_move_commands.contains(MoveCommands::Right) as i32 as f32
                - self.active_move_commands.contains(MoveCommands::Left) as i32 as f32)
                * right;
            translation *= timer.frame_delta().as_secs_f32() * self.translation_speed;
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

    pub fn fill_global_uniform_buffer(&self, aspect_ratio: f32) -> CameraUniformBufferContent {
        let right = self.direction.cross(self.rotational_up).normalize();
        let up = right.cross(self.direction).normalize();

        let view = cgmath::Matrix4::look_to_rh(self.position, self.direction, self.rotational_up);
        let projection = OPENGL_PROJECTION_TO_WGPU_PROJECTION * cgmath::perspective(VERTICAL_FOV, aspect_ratio, 0.01, 1000.0);
        let view_projection = projection * view;
        let inverse_projection = projection.invert().unwrap();
        //let inverse_view_projection = view_projection.invert().unwrap();

        let ndc_corner_camera = inverse_projection.transform_point(cgmath::point3(1.0, 1.0, 0.0));
        let ndc_camera_space_projected = cgmath::point2(-ndc_corner_camera.x / ndc_corner_camera.z, -ndc_corner_camera.y / ndc_corner_camera.z);

        CameraUniformBufferContent {
            view_projection,
            position: self.position.into(),
            right: right.into(),
            up: up.into(),
            direction: self.direction.into(),
            ndc_camera_space_projected: ndc_camera_space_projected.into(),
            tan_half_vertical_fov: (VERTICAL_FOV * 0.5).tan(),
            inv_tan_half_vertical_fov: 1.0 / (VERTICAL_FOV * 0.5).tan(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CameraUniformBufferContent {
    view_projection: cgmath::Matrix4<f32>,
    position: PaddedPoint3,
    right: PaddedVector3,
    up: PaddedVector3,
    direction: PaddedVector3,
    ndc_camera_space_projected: cgmath::Point2<f32>,
    tan_half_vertical_fov: f32,
    inv_tan_half_vertical_fov: f32,
}
