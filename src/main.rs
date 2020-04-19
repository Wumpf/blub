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
mod screen;
mod timer;
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

const TIMER_CONFIG: timer::TimeConfiguration = timer::TimeConfiguration::RealtimeRenderingFixedSimulationStep {
    simulation_delta: std::time::Duration::from_nanos((1000.0 * 1000.0 * 1000.0 / 120.0) as u64), // 120 simulation steps per second
    max_total_step_per_frame: std::time::Duration::from_nanos((1000.0 * 1000.0 * 1000.0 / 10.0) as u64), // stop catching up if slower than at 10fps
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
    glyph_brush: wgpu_glyph::GlyphBrush<'static, ()>,

    timer: timer::Timer,
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

        let hybrid_fluid = hybrid_fluid::HybridFluid::new(
            &device,
            wgpu::Extent3d {
                width: 128,
                height: 64,
                depth: 64,
            },
            2000000,
            &shader_dir,
            per_frame_resources.bind_group_layout(),
        );

        let particle_renderer =
            particle_renderer::ParticleRenderer::new(&device, &shader_dir, per_frame_resources.bind_group_layout(), &hybrid_fluid);

        let font = std::fs::read("assets/Roboto Mono.ttf").expect("loading font asset");
        let glyph_brush = wgpu_glyph::GlyphBrushBuilder::using_font_bytes(font)
            .expect("create glyph brush")
            .build(&device, Screen::FORMAT_BACKBUFFER);

        Application {
            window,
            window_surface,
            screen,

            device,
            command_queue,

            glyph_brush,
            shader_dir,
            particle_renderer,
            hybrid_fluid,

            camera: camera::Camera::new(),
            per_frame_resources,

            timer: timer::Timer::new(TIMER_CONFIG),
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
                        } => match virtual_keycode {
                            VirtualKeyCode::Escape => *control_flow = ControlFlow::Exit,
                            VirtualKeyCode::Space => self.restart_simulation(),
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
        });
    }

    pub fn restart_simulation(&mut self) {
        let mut init_encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Particle Init Encoder"),
        });

        self.hybrid_fluid.reset();
        self.hybrid_fluid.add_fluid_cube(
            &self.device,
            &mut init_encoder,
            cgmath::Point3::new(0.0, 0.0, 0.0),
            cgmath::Point3::new(64.0, 40.0, 64.0),
        );

        self.command_queue.submit(&[init_encoder.finish()]);
        self.device.poll(wgpu::Maintain::Wait);

        self.timer = timer::Timer::new(TIMER_CONFIG);
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

        // (GPU) Simulation.
        {
            let mut cpass = encoder.begin_compute_pass();
            cpass.set_bind_group(0, self.per_frame_resources.bind_group(), &[]);

            //  if self.timer.total_simulated_time().as_secs_f32() < 10.0 {
            while self.timer.simulation_step_loop() {
                self.hybrid_fluid.step(&mut cpass);
            }
            //    }
        }

        // Fluid drawing.
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
        // Text drawing.
        {
            let section = wgpu_glyph::Section {
                text: &format!(
                    "{:3.2}ms, FPS: {:3.2}\n\
                     rendered time  {:.2}s\n\
                     simulated time {:.2}s\n\
                     num simulation steps {}",
                    self.timer.frame_duration().as_secs_f64() * 1000.0,
                    1000.0 / 1000.0 / self.timer.frame_duration().as_secs_f64(),
                    self.timer.total_render_time().as_secs_f64(),
                    self.timer.total_simulated_time().as_secs_f64(),
                    self.timer.num_simulation_steps_performed_for_current_frame(),
                ),
                scale: wgpu_glyph::Scale::uniform(20.0),
                //color: [0.5, 0.5, 0.5, 1.0],
                ..wgpu_glyph::Section::default() // color, position, etc
            };
            self.glyph_brush.queue(section);
            self.glyph_brush
                .draw_queued(
                    &self.device,
                    &mut encoder,
                    &frame.view,
                    self.screen.resolution.width,
                    self.screen.resolution.height,
                )
                .expect("text drawing");
        }

        self.command_queue.submit(&[encoder.finish()]);

        std::mem::drop(frame);
        self.timer.on_frame_submitted();
    }
}

fn main() {
    env_logger::init_from_env(env_logger::Env::default().filter_or(env_logger::DEFAULT_FILTER_ENV, "warn,blub=info"));
    let event_loop = EventLoop::new();
    let mut application = futures::executor::block_on(Application::new(&event_loop));
    application.restart_simulation();
    application.run(event_loop);
}
