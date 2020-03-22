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

pub struct Screen {
    resolution: winit::dpi::PhysicalSize<u32>,
    swap_chain: wgpu::SwapChain,
    depth_view: wgpu::TextureView,
}

impl Screen {
    pub const FORMAT_BACKBUFFER: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;
    pub const FORMAT_DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    pub fn new(device: &wgpu::Device, window_surface: &wgpu::Surface, resolution: winit::dpi::PhysicalSize<u32>) -> Self {
        println!("creating screen with {:?}", resolution);

        let swap_chain = device.create_swap_chain(
            window_surface,
            &wgpu::SwapChainDescriptor {
                usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
                format: Self::FORMAT_BACKBUFFER,
                width: resolution.width,
                height: resolution.height,
                present_mode: wgpu::PresentMode::NoVsync,
            },
        );
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: resolution.width,
                height: resolution.height,
                depth: 1,
            },
            array_layer_count: 1,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::FORMAT_DEPTH,
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        });

        Screen {
            resolution,
            swap_chain,
            depth_view: depth_texture.create_default_view(),
        }
    }

    pub fn aspect_ratio(&self) -> f32 {
        self.resolution.width as f32 / self.resolution.height as f32
    }
}

pub struct Application {
    window: Window,
    window_surface: wgpu::Surface,
    screen: Screen,

    device: wgpu::Device,
    command_queue: wgpu::Queue,

    shader_dir: shader::ShaderDirectory,
    particle_renderer: particle_renderer::ParticleRenderer,

    camera: camera::Camera,
    ubo_camera: camera::CameraUniformBuffer,

    timer: rendertimer::RenderTimer,
}

impl Application {
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
        let screen = Screen::new(&device, &window_surface, window.inner_size());
        let backbuffer_resolution = window.inner_size();

        let shader_dir = shader::ShaderDirectory::new(Path::new("shader"));
        let ubo_camera = camera::CameraUniformBuffer::new(&device);
        let particle_renderer = particle_renderer::ParticleRenderer::new(&device, &shader_dir, &ubo_camera);

        Application {
            window,
            window_surface,
            screen,

            device,
            command_queue,

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

    fn window_resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        self.screen = Screen::new(&self.device, &self.window_surface, size);
    }

    fn update(&mut self) {
        if self.shader_dir.detected_change() {
            println!("reloading shaders...");
            self.particle_renderer.try_reload_shaders(&self.device, &self.shader_dir);
        }
        self.camera.update(&self.timer);
    }

    fn draw(&mut self) {
        let aspect_ratio = self.screen.aspect_ratio();
        let frame = self.screen.swap_chain.get_next_texture();
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });

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
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                    attachment: &self.screen.depth_view,
                    depth_load_op: wgpu::LoadOp::Clear,
                    depth_store_op: wgpu::StoreOp::Store,
                    stencil_load_op: wgpu::LoadOp::Clear,
                    stencil_store_op: wgpu::StoreOp::Store,
                    clear_depth: 1.0,
                    clear_stencil: 0,
                }),
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
