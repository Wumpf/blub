use std::path::Path;
use winit::{
    event::{Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
    window::WindowBuilder,
};

mod camera;
mod particle_renderer;
mod rendertimer;
mod shader;
mod uniformbuffer;

pub struct Application {
    window: Window,

    device: wgpu::Device,
    command_queue: wgpu::Queue,
    swap_chain: wgpu::SwapChain,

    window_surface: wgpu::Surface,
    backbuffer_resolution: winit::dpi::PhysicalSize<u32>,

    shader_dir: shader::ShaderDirectory,
    particle_renderer: particle_renderer::ParticleRenderer,

    camera: camera::Camera,
    ubo_camera: camera::CameraUniformBuffer,

    timer: rendertimer::RenderTimer,
}

impl Application {
    pub const FORMAT_BACKBUFFER: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;

    fn new(event_loop: &EventLoop<()>) -> Application {
        let window = WindowBuilder::new()
            .with_title("Blub")
            .with_resizable(true)
            .with_inner_size(winit::dpi::LogicalSize::new(1980, 1080))
            .build(&event_loop)
            .unwrap();

        let adapter = wgpu::Adapter::request(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            backends: wgpu::BackendBit::PRIMARY,
        })
        .unwrap();

        let (device, command_queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            extensions: wgpu::Extensions { anisotropic_filtering: true },
            limits: wgpu::Limits::default(),
        });

        let window_surface = wgpu::Surface::create(&window);
        let backbuffer_resolution = window.inner_size();
        let swap_chain = device.create_swap_chain(&window_surface, &Self::swap_chain_desc(window.inner_size()));

        let shader_dir = shader::ShaderDirectory::new(Path::new("shader"));
        let ubo_camera = camera::CameraUniformBuffer::new(&device);
        let particle_renderer = particle_renderer::ParticleRenderer::new(&device, &shader_dir, &ubo_camera);

        Application {
            window,
            device,
            command_queue,
            swap_chain,

            window_surface,
            backbuffer_resolution,

            shader_dir,
            particle_renderer,

            camera: camera::Camera::new(),
            ubo_camera,

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

    fn swap_chain_desc(size: winit::dpi::PhysicalSize<u32>) -> wgpu::SwapChainDescriptor {
        wgpu::SwapChainDescriptor {
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            format: Self::FORMAT_BACKBUFFER,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::NoVsync,
        }
    }

    fn window_resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        println!("resizing screen to {:?}", size);
        self.backbuffer_resolution = size;
        self.swap_chain = self.device.create_swap_chain(&self.window_surface, &Self::swap_chain_desc(size));
    }

    fn update(&mut self) {
        if self.shader_dir.detected_change() {
            println!("reloading shaders...");
            self.particle_renderer.try_reload_shaders(&self.device, &self.shader_dir);
        }
        self.camera.update(&self.timer);
    }

    fn draw(&mut self) {
        let frame = self.swap_chain.get_next_texture();
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });

        let aspect_ratio = self.backbuffer_resolution.width as f32 / self.backbuffer_resolution.height as f32;
        self.ubo_camera
            .update_content(&mut encoder, &self.device, self.camera.fill_uniform_buffer(aspect_ratio));

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
                depth_stencil_attachment: None,
            });

            self.particle_renderer.draw(&mut rpass);
        }
        self.command_queue.submit(&[encoder.finish()]);

        std::mem::drop(frame);
        self.timer.on_frame_submitted();
    }
}

fn main() {
    let event_loop = EventLoop::new();
    let application = Application::new(&event_loop);
    application.run(event_loop);
}
