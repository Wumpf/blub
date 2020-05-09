#[macro_use]
extern crate more_asserts;
#[macro_use]
extern crate log;

mod camera;
mod gui;
mod hybrid_fluid;
mod particle_renderer;
mod per_frame_resources;
mod scene;
mod screen;
mod simulation_controller;
mod static_line_renderer;
mod timer;
mod wgpu_utils;

use per_frame_resources::*;
use screen::*;
use simulation_controller::SimulationControllerStatus;
use std::path::{Path, PathBuf};
use wgpu_utils::shader;
use winit::{
    event::{Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
    window::WindowBuilder,
};

struct Application {
    window: Window,
    window_surface: wgpu::Surface,
    screen: Screen,
    scheduled_screenshot: PathBuf,

    device: wgpu::Device,
    command_queue: wgpu::Queue,

    shader_dir: shader::ShaderDirectory,
    scene: scene::Scene,
    scene_renderer: scene::SceneRenderer,
    simulation_controller: simulation_controller::SimulationController,
    gui: gui::GUI,

    camera: camera::Camera,
    per_frame_resources: PerFrameResources,
}

impl Application {
    async fn new(event_loop: &EventLoop<()>) -> Application {
        let window = WindowBuilder::new()
            .with_title("Blub")
            .with_resizable(true)
            .with_inner_size(winit::dpi::LogicalSize::new(1980, 1080))
            .build(&event_loop)
            .unwrap();

        let window_surface = wgpu::Surface::create(&window);
        let adapter = wgpu::Adapter::request(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&window_surface),
            },
            // Didn't get it to work with DX12 so far: Issues with bindings in compute pipelines.
            // Some (!) of these issues are fixed with https://github.com/gfx-rs/wgpu/pull/572
            wgpu::BackendBit::PRIMARY, //wgpu::BackendBit::DX12,
        )
        .await
        .unwrap();

        let (device, mut command_queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                extensions: wgpu::Extensions { anisotropic_filtering: true },
                limits: wgpu::Limits::default(),
            })
            .await;

        let shader_dir = shader::ShaderDirectory::new(Path::new("shader"));

        let screen = Screen::new(&device, &window_surface, window.inner_size(), &shader_dir);
        let per_frame_resources = PerFrameResources::new(&device);

        let mut init_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Startup Encoder"),
        });

        let scene = scene::Scene::new(&device, &mut init_encoder, &shader_dir, per_frame_resources.bind_group_layout());
        let simulation_controller = simulation_controller::SimulationController::new();
        let mut scene_renderer = scene::SceneRenderer::new(&device, &shader_dir, per_frame_resources.bind_group_layout());
        scene_renderer.on_new_scene(&device, &mut init_encoder, &scene);

        let gui = gui::GUI::new(&device, &window, &mut command_queue);

        command_queue.submit(&[init_encoder.finish()]);
        device.poll(wgpu::Maintain::Wait);

        Application {
            window,
            window_surface,
            screen,
            scheduled_screenshot: PathBuf::default(),

            device,
            command_queue,

            shader_dir,
            scene,
            scene_renderer,
            simulation_controller,
            gui,

            camera: camera::Camera::new(),
            per_frame_resources,
        }
    }

    fn schedule_screenshot(&mut self) {
        for i in 0..usize::MAX {
            let screenshot = PathBuf::from(format!("screenshot{}.png", i));
            if !screenshot.exists() {
                self.scheduled_screenshot = screenshot;
                return;
            }
        }
    }

    fn run(mut self, event_loop: EventLoop<()>) {
        event_loop.run(move |event, _, control_flow| {
            // ControlFlow::Poll continuously runs the event loop, even if the OS hasn't
            // dispatched any events. This is ideal for games and similar applications.
            *control_flow = ControlFlow::Poll;

            match &event {
                Event::WindowEvent { event, .. } => {
                    self.camera.on_window_event(&event);
                    match event {
                        WindowEvent::CloseRequested => {
                            *control_flow = ControlFlow::Exit;
                        }
                        WindowEvent::Resized(size) => {
                            self.window_resize(*size);
                        }
                        WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                            self.window_resize(**new_inner_size);
                        }
                        WindowEvent::KeyboardInput {
                            input:
                                KeyboardInput {
                                    state,
                                    virtual_keycode: Some(virtual_keycode),
                                    ..
                                },
                            ..
                        } => match virtual_keycode {
                            VirtualKeyCode::Escape => *control_flow = ControlFlow::Exit,
                            VirtualKeyCode::Snapshot => self.schedule_screenshot(), // Bug? doesn't seem to receive a winit::event::ElementState::Pressed event.
                            VirtualKeyCode::Space => {
                                if let winit::event::ElementState::Pressed = state {
                                    self.simulation_controller.status = if self.simulation_controller.status == SimulationControllerStatus::Paused {
                                        SimulationControllerStatus::Realtime
                                    } else {
                                        SimulationControllerStatus::Paused
                                    }
                                }
                            }
                            _ => {}
                        },
                        _ => {}
                    }
                }
                Event::DeviceEvent { event, .. } => {
                    self.camera.on_device_event(&event);
                }
                Event::MainEventsCleared => {
                    self.update();
                    self.window.request_redraw();
                }
                Event::RedrawRequested(_) => {
                    self.draw();
                }
                _ => (),
            }

            self.gui.handle_event(&self.window, &event);
        });
    }

    fn window_resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        // occasionally window size drops to zero which causes crashes along the way
        if self.screen.resolution != size && size.width != 0 && size.height != 0 {
            self.screen = Screen::new(&self.device, &self.window_surface, size, &self.shader_dir);
        }
    }

    fn update(&mut self) {
        if self.shader_dir.detected_change() {
            info!("reloading shaders...");
            self.scene_renderer.try_reload_shaders(&self.device, &self.shader_dir);
            self.scene.try_reload_shaders(&self.device, &self.shader_dir);
        }
        self.camera.update(self.simulation_controller.timer());

        if self.simulation_controller.handle_scheduled_restart() {
            // Idiot proof way to reset the fluid: Recreate everything.
            // Previously, we reset the particles but then previous pressure computation results crept in making the reset more undeterministic than necessary
            // Note that it is NOT deterministic due to some parallel processes reordering floating point operations at random.
            // TODO?: Keeps old scene alive until new one is fully set.
            let mut init_encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Reset Scene Encoder"),
            });
            self.scene = scene::Scene::new(
                &self.device,
                &mut init_encoder,
                &self.shader_dir,
                self.per_frame_resources.bind_group_layout(),
            );
            self.scene_renderer.on_new_scene(&self.device, &mut init_encoder, &self.scene);
            self.command_queue.submit(&[init_encoder.finish()]);
            self.device.poll(wgpu::Maintain::Wait);
        }

        self.simulation_controller.fast_forward_steps(
            &self.device,
            &self.command_queue,
            &mut self.scene,
            self.per_frame_resources.bind_group(), // values from last draw are good enough.
        );

        if let simulation_controller::SimulationControllerStatus::Record { output_directory } = &self.simulation_controller.status {
            self.scheduled_screenshot = output_directory.join(format!("{}.png", self.simulation_controller.timer().num_frames_rendered()));
        }
    }

    fn draw(&mut self) {
        let aspect_ratio = self.screen.aspect_ratio();
        let frame = self.screen.start_frame();
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Encoder: Frame Main"),
        });

        self.per_frame_resources
            .update_gpu_data(&mut encoder, &self.device, &self.camera, self.simulation_controller.timer(), aspect_ratio);

        self.simulation_controller
            .frame_steps(&self.scene, &mut encoder, self.per_frame_resources.bind_group());
        self.scene_renderer.draw(
            &self.scene,
            &mut encoder,
            self.screen.backbuffer(),
            self.screen.depthbuffer(),
            self.per_frame_resources.bind_group(),
        );

        if self.scheduled_screenshot != PathBuf::default() {
            self.screen.take_screenshot(&mut encoder, &self.scheduled_screenshot);
            self.scheduled_screenshot = PathBuf::default();
        }

        self.gui.draw(
            &self.device,
            &self.window,
            &mut encoder,
            &self.screen.backbuffer(),
            &mut self.simulation_controller,
        );

        self.screen.copy_to_swapchain(&frame, &mut encoder);
        self.command_queue.submit(&[encoder.finish()]);
        self.screen.end_frame(&self.device, frame);
        self.simulation_controller.on_frame_submitted();
    }
}

fn main() {
    env_logger::init_from_env(env_logger::Env::default().filter_or(env_logger::DEFAULT_FILTER_ENV, "warn,blub=info"));
    let event_loop = EventLoop::new();
    let application = futures::executor::block_on(Application::new(&event_loop));
    application.run(event_loop);
}
