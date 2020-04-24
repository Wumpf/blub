use crate::simulation_controller::{SimulationController, SimulationControllerStatus};
use imgui::im_str;

pub struct GUI {
    imgui_context: imgui::Context,
    imgui_platform: imgui_winit_support::WinitPlatform,
    imgui_renderer: imgui_wgpu::Renderer,
}

impl GUI {
    pub fn new(device: &wgpu::Device, window: &winit::window::Window, command_queue: &mut wgpu::Queue) -> Self {
        let mut imgui_context = imgui::Context::create();

        let mut imgui_platform = imgui_winit_support::WinitPlatform::init(&mut imgui_context);
        imgui_platform.attach_window(imgui_context.io_mut(), &window, imgui_winit_support::HiDpiMode::Default);

        imgui_context.set_ini_filename(None);

        let hidpi_factor = 1.0;
        let font_size = (13.0 * hidpi_factor) as f32;
        imgui_context.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;
        imgui_context.fonts().add_font(&[imgui::FontSource::DefaultFontData {
            config: Some(imgui::FontConfig {
                oversample_h: 1,
                pixel_snap_h: true,
                size_pixels: font_size,
                ..Default::default()
            }),
        }]);

        let imgui_renderer = imgui_wgpu::Renderer::new(&mut imgui_context, &device, command_queue, crate::Screen::FORMAT_BACKBUFFER, None);

        GUI {
            imgui_context,
            imgui_platform,
            imgui_renderer,
        }
    }

    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        window: &winit::window::Window,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        simulation_controller: &mut SimulationController,
    ) {
        let context = &mut self.imgui_context;
        //let state = &mut self.state;
        //self.imgui_context.io_mut().update_delta_time(timer.frame_delta()); // Needed?
        self.imgui_platform
            .prepare_frame(context.io_mut(), window)
            .expect("Failed to prepare imgui frame");
        let ui = context.frame();
        {
            let window = imgui::Window::new(im_str!("Blub"));
            window
                .position([0.0, 0.0], imgui::Condition::FirstUseEver)
                .always_auto_resize(true)
                .build(&ui, || {
                    ui.text(im_str!(
                        "{:3.2}ms, FPS: {:3.2}",
                        simulation_controller.timer.duration_for_last_frame().as_secs_f64() * 1000.0,
                        1000.0 / 1000.0 / simulation_controller.timer.duration_for_last_frame().as_secs_f64()
                    ));
                    ui.separator();
                    ui.text(im_str!(
                        "num simulation steps current frame: {}",
                        simulation_controller.timer.num_simulation_steps_performed_for_current_frame()
                    ));
                    ui.text(im_str!(
                        "rendered time:  {:.2}",
                        simulation_controller.timer.total_render_time().as_secs_f64()
                    ));
                    ui.text(im_str!(
                        "simulated time: {:.2}",
                        simulation_controller.timer.total_simulated_time().as_secs_f64()
                    ));
                    ui.text(im_str!(
                        "total num simulation steps: {}",
                        simulation_controller.timer.num_simulation_steps_performed()
                    ));

                    ui.separator();

                    let mut simulation_time_seconds = simulation_controller.simulation_length.as_secs_f32();
                    if ui
                        .input_float(im_str!("target simulation time (s)"), &mut simulation_time_seconds)
                        .step(0.1)
                        .enter_returns_true(true)
                        .build()
                    {
                        simulation_controller.simulation_length = std::time::Duration::from_secs_f32(simulation_time_seconds);
                    }

                    if ui.small_button(im_str!("Reset (Space)")) {
                        simulation_controller.scheduled_restart = true;
                    }
                    ui.same_line(0.0);
                    if simulation_controller.status == SimulationControllerStatus::Paused {
                        if ui.small_button(im_str!("Continue")) {
                            simulation_controller.status = SimulationControllerStatus::Realtime;
                        }
                    } else {
                        if ui.small_button(im_str!("Pause")) {
                            simulation_controller.status = SimulationControllerStatus::Paused;
                        }
                    }
                });
        }

        self.imgui_platform.prepare_render(&ui, &window);
        self.imgui_renderer
            .render(ui.render(), &device, encoder, view)
            .expect("IMGUI rendering failed");
    }

    pub fn handle_event(&mut self, window: &winit::window::Window, event: &winit::event::Event<()>) {
        self.imgui_platform.handle_event(self.imgui_context.io_mut(), window, event);
    }
}
