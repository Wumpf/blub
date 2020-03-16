use winit::{
    event::{Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
    window::Window,
};

struct Application {
    device: wgpu::Device,
    command_queue: wgpu::Queue,
    swap_chain: wgpu::SwapChain,
    window_surface: wgpu::Surface,
}

impl Application {

    fn new(window: &Window) -> Application {
        let adapter = wgpu::Adapter::request(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                backends: wgpu::BackendBit::PRIMARY,
            },
        )
        .unwrap();

        let (device, command_queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            extensions: wgpu::Extensions {
                anisotropic_filtering: true,
            },
            limits: wgpu::Limits::default(),
        });


        let window_surface = wgpu::Surface::create(window);
        let swap_chain = device.create_swap_chain(&window_surface, &Self::swap_chain_desc(window.inner_size()));

        Application {
            device,
            command_queue,
            swap_chain,
            window_surface,
        }
    }

    fn swap_chain_desc(size: winit::dpi::PhysicalSize<u32>) -> wgpu::SwapChainDescriptor {
        wgpu::SwapChainDescriptor {
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            format: wgpu::TextureFormat::Bgra8Unorm,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::NoVsync,
        }
    }

    fn window_resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        self.swap_chain = self.device.create_swap_chain(&self.window_surface, &Self::swap_chain_desc(size));
    }

    fn update(&mut self) {

    }

    fn draw(&mut self) {
        let frame = self.swap_chain.get_next_texture();
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor{
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
        }
        self.command_queue.submit(&[encoder.finish()]);
    }
}

fn main() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Blub")
        .with_resizable(true)
        .with_inner_size(winit::dpi::LogicalSize::new(1980, 1080))
        .build(&event_loop)
        .unwrap();

    let mut application = Application::new(&window);

    event_loop.run(move |event, _, control_flow| {
        // ControlFlow::Poll continuously runs the event loop, even if the OS hasn't
        // dispatched any events. This is ideal for games and similar applications.
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent { event, .. } => {
                match event {
                    WindowEvent::CloseRequested => {
                        *control_flow = ControlFlow::Exit;
                    }
                    WindowEvent::Resized(size) => {
                        application.window_resize(size);
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
            Event::MainEventsCleared => {
                application.update();
                window.request_redraw();
            }
            Event::RedrawRequested(_) => {
                application.draw();
            }
            _ => (),
        }
    });
}
