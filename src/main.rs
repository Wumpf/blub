#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate more_asserts;
#[macro_use]
extern crate log;

mod camera;
mod hybrid_fluid;
mod particle_renderer;
mod per_frame_resources;
mod rendertimer;
mod screen;
mod wgpu_utils;

use per_frame_resources::*;
use screen::*;
use std::path::Path;
use wgpu_utils::shader;
use winit::{
    event::{Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
    window::WindowBuilder,
};

pub struct Application {
    window: Window,
    window_surface: wgpu::Surface,
    screen: Screen,

    device: wgpu::Device,
    command_queue: wgpu::Queue,

    shader_dir: shader::ShaderDirectory,
    particle_renderer: particle_renderer::ParticleRenderer,
    hybrid_fluid: hybrid_fluid::HybridFluid,

    camera: camera::Camera,
    per_frame_resources: PerFrameResources,

    timer: rendertimer::RenderTimer,
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

        let (device, command_queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                extensions: wgpu::Extensions { anisotropic_filtering: true },
                limits: wgpu::Limits::default(),
            })
            .await;

        let screen = Screen::new(&device, &window_surface, window.inner_size());

        let shader_dir = shader::ShaderDirectory::new(Path::new("shader"));
        let per_frame_resources = PerFrameResources::new(&device);

        let mut init_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Init Encoder") });

        let mut hybrid_fluid = hybrid_fluid::HybridFluid::new(
            &device,
            wgpu::Extent3d {
                width: 128,
                height: 64,
                depth: 64,
            },
            1000000,
            &shader_dir,
            per_frame_resources.bind_group_layout(),
        );
        hybrid_fluid.add_fluid_cube(
            &device,
            &mut init_encoder,
            cgmath::Point3::new(2.0, 2.0, 2.0),
            cgmath::Point3::new(32.0, 62.0, 62.0),
        );

        let particle_renderer =
            particle_renderer::ParticleRenderer::new(&device, &shader_dir, per_frame_resources.bind_group_layout(), &hybrid_fluid);

        command_queue.submit(&[init_encoder.finish()]);

        Application {
            window,
            window_surface,
            screen,

            device,
            command_queue,

            shader_dir,
            particle_renderer,
            hybrid_fluid,

            camera: camera::Camera::new(),
            per_frame_resources,

            timer: rendertimer::RenderTimer::new(),
        }
    }

    fn run(mut self, event_loop: EventLoop<()>) {
        event_loop.run(move |event, _, control_flow| {
            // ControlFlow::Poll continuously runs the event loop, even if the OS hasn't
            // dispatched any events. This is ideal for games and similar applications.
            *control_flow = ControlFlow::Poll;

            match event {
                Event::WindowEvent { event, .. } => {
                    self.camera.on_window_event(&event);
                    match event {
                        WindowEvent::CloseRequested => {
                            *control_flow = ControlFlow::Exit;
                        }
                        WindowEvent::Resized(size) => {
                            self.window_resize(size);
                        }
                        WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                            self.window_resize(*new_inner_size);
                        }
                        WindowEvent::KeyboardInput {
                            input:
                                KeyboardInput {
                                    virtual_keycode: Some(virtual_keycode),
                                    ..
                                },
                            ..
                        } => {
                            if virtual_keycode == VirtualKeyCode::Escape {
                                *control_flow = ControlFlow::Exit;
                            }
                        }
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
        });
    }

    fn window_resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        // occasionally window size drops to zero which causes crashes along the way
        if self.screen.resolution != size && size.width != 0 && size.height != 0 {
            self.screen = Screen::new(&self.device, &self.window_surface, size);
        }
    }

    fn update(&mut self) {
        if self.shader_dir.detected_change() {
            info!("reloading shaders...");
            self.particle_renderer.try_reload_shaders(&self.device, &self.shader_dir);
            self.hybrid_fluid.try_reload_shaders(&self.device, &self.shader_dir);
        }
        self.camera.update(&self.timer);
    }

    fn draw(&mut self) {
        let aspect_ratio = self.screen.aspect_ratio();
        let (frame, depth_view) = self.screen.get_next_frame();
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Encoder: Frame Main"),
        });

        self.per_frame_resources
            .update_gpu_data(&mut encoder, &self.device, &self.camera, &self.timer, aspect_ratio);

        {
            let mut cpass = encoder.begin_compute_pass();
            cpass.set_bind_group(0, self.per_frame_resources.bind_group(), &[]);
            self.hybrid_fluid.step(&mut cpass);
        }

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &frame.view,
                    resolve_target: None,
                    load_op: wgpu::LoadOp::Clear,
                    store_op: wgpu::StoreOp::Store,
                    clear_color: wgpu::Color {
                        r: 0.1,
                        g: 0.2,
                        b: 0.3,
                        a: 1.0,
                    },
                }],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                    attachment: depth_view,
                    depth_load_op: wgpu::LoadOp::Clear,
                    depth_store_op: wgpu::StoreOp::Store,
                    stencil_load_op: wgpu::LoadOp::Clear,
                    stencil_store_op: wgpu::StoreOp::Store,
                    clear_depth: 1.0,
                    clear_stencil: 0,
                }),
            });

            rpass.set_bind_group(0, self.per_frame_resources.bind_group(), &[]);
            self.particle_renderer.draw(&mut rpass, self.hybrid_fluid.num_particles());
        }
        self.command_queue.submit(&[encoder.finish()]);

        std::mem::drop(frame);
        self.timer.on_frame_submitted();
    }
}

fn main() {
    env_logger::init_from_env(env_logger::Env::default().filter_or(env_logger::DEFAULT_FILTER_ENV, "warn,blub=info"));
    let event_loop = EventLoop::new();
    let application = futures::executor::block_on(Application::new(&event_loop));
    application.run(event_loop);
}
