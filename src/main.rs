use winit::{
    event::{Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

fn main() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Blub")
        .with_resizable(true)
        .with_inner_size(winit::dpi::LogicalSize::new(1980, 1080))
        .build(&event_loop)
        .unwrap();

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
                    WindowEvent::Resized(_) => {
                        // sc_desc.width = size.width;
                        // sc_desc.height = size.height;
                        // swap_chain = device.create_swap_chain(&surface, &sc_desc);
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
                update();
                window.request_redraw();
            }
            Event::RedrawRequested(_) => {
                draw();
            }
            _ => (),
        }
    });
}

fn update() {}

fn draw() {}
