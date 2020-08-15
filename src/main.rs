#[macro_use]
extern crate more_asserts;
#[macro_use]
extern crate log;
#[macro_use]
extern crate strum_macros;

mod camera;
mod gui;
mod per_frame_resources;
mod render_output;
mod renderer;
mod scene;
mod simulation;
mod simulation_controller;
mod timer;
mod wgpu_utils;

use per_frame_resources::*;
use render_output::{hdr_backbuffer::HdrBackbuffer, screen::Screen, screenshot_recorder::ScreenshotRecorder};
use renderer::SceneRenderer;
use simulation_controller::SimulationControllerStatus;
use std::{
    path::{Path, PathBuf},
    time::Duration,
};
use wgpu_utils::{pipelines, shader};
use winit::{
    event::{Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop, EventLoopProxy},
    window::Window,
    window::WindowBuilder,
};

#[derive(Debug, Clone)]
pub enum ApplicationEvent {
    LoadScene(PathBuf),
    ResetScene,
    FastForwardSimulation(Duration),
    ResetAndStartRecording { recording_fps: f64 }, // to stop recording, pause the simulation controller.
    ChangePresentMode(wgpu::PresentMode),
}

struct Application {
    window: Window,
    window_surface: wgpu::Surface,
    screen: Screen,
    hdr_backbuffer: HdrBackbuffer,
    screenshot_recorder: ScreenshotRecorder,

    device: wgpu::Device,
    command_queue: wgpu::Queue,

    shader_dir: shader::ShaderDirectory,
    pipeline_manager: pipelines::PipelineManager,
    scene: Option<scene::Scene>,
    scene_renderer: SceneRenderer,
    simulation_controller: simulation_controller::SimulationController,
    gui: gui::GUI,

    camera: camera::Camera,
    per_frame_resources: PerFrameResources,
}

impl Application {
    async fn new(event_loop: &EventLoop<ApplicationEvent>) -> Application {
        let wgpu_instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY); //wgpu::BackendBit::DX12);
        let window = WindowBuilder::new()
            .with_title("Blub")
            .with_resizable(true)
            .with_inner_size(winit::dpi::LogicalSize::new(1980, 1080))
            .build(&event_loop)
            .unwrap();

        let window_surface = unsafe { wgpu_instance.create_surface(&window) };
        let adapter = wgpu_instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&window_surface),
            })
            .await
            .unwrap();

        let (device, mut command_queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::PUSH_CONSTANTS,
                    limits: wgpu::Limits {
                        max_storage_textures_per_shader_stage: 16, // Doesn't need to be as many https://github.com/gfx-rs/wgpu/pull/798
                        max_push_constant_size: 8,
                        ..Default::default()
                    },
                    shader_validation: false, // Disabled shader validation for now since we use too many things that it doesn't know about.
                },
                None, //Some(Path::new("C:/dev/blub/trace")),
            )
            .await
            .unwrap();

        let shader_dir = shader::ShaderDirectory::new(Path::new("shader"));
        let mut pipeline_manager = pipelines::PipelineManager::new();

        let screen = Screen::new(&device, &window_surface, Screen::DEFAULT_PRESENT_MODE, window.inner_size(), &shader_dir);
        let hdr_backbuffer = HdrBackbuffer::new(&device, &screen, &shader_dir);
        let per_frame_resources = PerFrameResources::new(&device);
        let simulation_controller = simulation_controller::SimulationController::new();
        let scene_renderer = SceneRenderer::new(
            &device,
            &shader_dir,
            &mut pipeline_manager,
            per_frame_resources.bind_group_layout(),
            &hdr_backbuffer,
        );
        let gui = gui::GUI::new(&device, &window, &mut command_queue);

        Application {
            window,
            window_surface,
            screen,
            hdr_backbuffer,
            screenshot_recorder: ScreenshotRecorder::new(),

            device,
            command_queue,

            shader_dir,
            pipeline_manager,
            scene: None,
            scene_renderer,
            simulation_controller,
            gui,

            camera: camera::Camera::new(),
            per_frame_resources,
        }
    }

    pub fn load_scene(&mut self, scene_path: &Path) {
        let new_scene = scene::Scene::new(
            scene_path,
            &self.device,
            &self.command_queue,
            &self.shader_dir,
            &mut self.pipeline_manager,
            self.per_frame_resources.bind_group_layout(),
        );

        match new_scene {
            Ok(scene) => {
                self.scene_renderer.on_new_scene(&self.command_queue, &scene);
                self.scene = Some(scene);
            }
            Err(error) => {
                error!("Failed to load scene from {:?}: {:?}", scene_path, error);
            }
        }
    }

    fn run(mut self, event_loop: EventLoop<ApplicationEvent>) {
        let event_loop_proxy = event_loop.create_proxy();

        event_loop.run(move |event, _, control_flow| {
            // ControlFlow::Poll continuously runs the event loop, even if the OS hasn't
            // dispatched any events. This is ideal for games and similar applications.
            *control_flow = ControlFlow::Poll;

            match &event {
                Event::UserEvent(event) => match event {
                    ApplicationEvent::LoadScene(scene_path) => {
                        self.load_scene(scene_path);
                        self.simulation_controller.restart();
                    }
                    ApplicationEvent::ResetScene => {
                        if let Some(ref mut scene) = self.scene {
                            scene.reset(
                                &self.device,
                                &self.command_queue,
                                &self.shader_dir,
                                &mut self.pipeline_manager,
                                self.per_frame_resources.bind_group_layout(),
                            );
                        }
                        self.simulation_controller.restart();
                    }
                    ApplicationEvent::FastForwardSimulation(simulation_jump_length) => {
                        if let Some(ref mut scene) = self.scene {
                            self.simulation_controller.fast_forward_steps(
                                *simulation_jump_length,
                                &self.device,
                                &self.command_queue,
                                scene,
                                &self.pipeline_manager,
                                self.per_frame_resources.bind_group(), // values from last draw are good enough.
                            );
                        }
                    }
                    ApplicationEvent::ResetAndStartRecording { recording_fps } => {
                        if let Some(ref mut scene) = self.scene {
                            scene.reset(
                                &self.device,
                                &self.command_queue,
                                &self.shader_dir,
                                &mut self.pipeline_manager,
                                self.per_frame_resources.bind_group_layout(),
                            );
                        }
                        self.simulation_controller.restart();
                        self.simulation_controller.start_recording_with_fixed_frame_length(*recording_fps);
                        self.screenshot_recorder.start_next_recording();
                    }
                    ApplicationEvent::ChangePresentMode(present_mode) => {
                        self.screen = Screen::new(
                            &self.device,
                            &self.window_surface,
                            *present_mode,
                            self.screen.resolution(),
                            &self.shader_dir,
                        );
                    }
                },
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
                            VirtualKeyCode::Snapshot => self.screenshot_recorder.schedule_next_screenshot(), // Bug? doesn't seem to receive a winit::event::ElementState::Pressed event.
                            VirtualKeyCode::Space => {
                                if let winit::event::ElementState::Pressed = state {
                                    self.simulation_controller.pause_or_resume();
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
                    self.draw(&event_loop_proxy);
                }
                Event::LoopDestroyed => {
                    // workaround for errors on shutdown while recording screenshots
                    self.screen.wait_for_pending_screenshots(&self.device);
                }
                _ => (),
            }

            self.gui.handle_event(&self.window, &event);
        });
    }

    fn window_resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        // occasionally window size drops to zero which causes crashes along the way
        if self.screen.resolution() != size && size.width != 0 && size.height != 0 {
            self.screen = Screen::new(&self.device, &self.window_surface, self.screen.present_mode(), size, &self.shader_dir);
            self.hdr_backbuffer = HdrBackbuffer::new(&self.device, &self.screen, &self.shader_dir);
            self.scene_renderer.on_window_resize(&self.device, &self.hdr_backbuffer);
        }
    }

    fn update(&mut self) {
        if self.shader_dir.detected_change() {
            info!("reloading shaders...");
            self.pipeline_manager.reload_all(&self.device, &self.shader_dir);
        }
        self.camera.update(self.simulation_controller.timer());

        if self.simulation_controller.status() == SimulationControllerStatus::Paused {
            self.screenshot_recorder.stop_recording();
        }
    }

    fn draw(&mut self, event_loop_proxy: &EventLoopProxy<ApplicationEvent>) {
        let aspect_ratio = self.screen.aspect_ratio();
        let frame = self.screen.start_frame();
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Encoder: Frame Main"),
        });

        if let Some(ref mut scene) = self.scene {
            self.per_frame_resources.update_gpu_data(
                &self.command_queue,
                self.camera.fill_global_uniform_buffer(aspect_ratio),
                self.simulation_controller.timer().fill_global_uniform_buffer(),
                self.scene_renderer.fill_global_uniform_buffer(&scene),
            );

            self.simulation_controller.frame_steps(
                scene,
                &mut encoder,
                &self.pipeline_manager,
                &self.command_queue,
                self.per_frame_resources.bind_group(),
            );
            self.scene_renderer.draw(
                scene,
                &mut encoder,
                &self.pipeline_manager,
                self.hdr_backbuffer.texture_view(),
                self.screen.depthbuffer(),
                self.per_frame_resources.bind_group(),
            );
        }

        self.hdr_backbuffer.tonemap(&self.screen.backbuffer(), &mut encoder);

        self.screenshot_recorder.capture_screenshot(&mut self.screen, &self.device, &mut encoder);

        self.gui.draw(
            &self.device,
            &self.window,
            &mut encoder,
            &self.command_queue,
            &self.screen.backbuffer(),
            &mut self.simulation_controller,
            &mut self.scene_renderer,
            event_loop_proxy,
        );

        self.screen.copy_to_swapchain(&frame, &mut encoder);
        self.command_queue.submit(Some(encoder.finish()));
        self.screen.end_frame(frame);
        self.simulation_controller.on_frame_submitted();
    }
}

fn main() {
    // Silence warnings from `naga::front::spirv` for now since as of writing it doesn't know enough spirv yet.
    env_logger::init_from_env(env_logger::Env::default().filter_or(env_logger::DEFAULT_FILTER_ENV, "warn,blub=info,naga=error"));
    let event_loop = EventLoop::<ApplicationEvent>::with_user_event();
    let application = futures::executor::block_on(Application::new(&event_loop));
    application.run(event_loop);
}
