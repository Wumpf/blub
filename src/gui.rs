use crate::simulation_controller::{SimulationController, SimulationControllerStatus};
use imgui::im_str;
use std::{path::PathBuf, time::Duration};

struct GUIState {
    fast_forward_length_seconds: f32,
}
pub struct GUI {
    imgui_context: imgui::Context,
    imgui_platform: imgui_winit_support::WinitPlatform,
    imgui_renderer: imgui_wgpu::Renderer,

    state: GUIState,
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

            state: GUIState {
                fast_forward_length_seconds: 5.0,
            },
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
        let state = &mut self.state;

        //let state = &mut self.state;
        //self.imgui_context.io_mut().update_delta_time(tim().frame_delta()); // Needed?
        self.imgui_platform
            .prepare_frame(context.io_mut(), window)
            .expect("Failed to prepare imgui frame");
        let ui = context.frame();
        {
            const DEFAULT_BUTTON_HEIGHT: f32 = 19.0;

            let window = imgui::Window::new(im_str!("Blub"));
            window
                .position([0.0, 0.0], imgui::Condition::FirstUseEver)
                .always_auto_resize(true)
                .build(&ui, || {
                    ui.text(im_str!(
                        "{:3.2}ms, FPS: {:3.2}",
                        simulation_controller.timer().duration_last_frame().as_secs_f64() * 1000.0,
                        1000.0 / 1000.0 / simulation_controller.timer().duration_last_frame().as_secs_f64()
                    ));
                    ui.plot_histogram(
                        im_str!(""),
                        &simulation_controller
                            .timer()
                            .duration_last_frame_history()
                            .iter()
                            .map(|duration| duration.as_secs_f32())
                            .collect::<Vec<f32>>(),
                    )
                    .scale_min(0.0)
                    .graph_size([300.0, 40.0])
                    .build();
                    ui.separator();
                    ui.text(im_str!(
                        "num simulation steps current frame: {}",
                        simulation_controller.timer().num_simulation_steps_performed_for_current_frame()
                    ));
                    ui.text(im_str!(
                        "rendered time:  {:.2}",
                        simulation_controller.timer().total_render_time().as_secs_f64()
                    ));
                    ui.text(im_str!(
                        "simulated time: {:.2}",
                        simulation_controller.timer().total_simulated_time().as_secs_f64()
                    ));
                    ui.text(im_str!(
                        "total num simulation steps: {}",
                        simulation_controller.timer().num_simulation_steps_performed()
                    ));

                    ui.separator();

                    let mut simulation_time_seconds = simulation_controller.simulation_stop_time.as_secs_f32();
                    ui.push_item_width(110.0);
                    if ui
                        .input_float(im_str!("target simulation time (s)"), &mut simulation_time_seconds)
                        .step(0.1)
                        .enter_returns_true(true)
                        .build()
                    {
                        simulation_controller.simulation_stop_time = std::time::Duration::from_secs_f32(simulation_time_seconds);
                    }

                    let mut simulation_steps_per_second = simulation_controller.simulation_steps_per_second() as i32;
                    if ui
                        .input_int(im_str!("simulation steps per second"), &mut simulation_steps_per_second)
                        .step(10)
                        .enter_returns_true(true)
                        .build()
                    {
                        simulation_controller.set_simulation_steps_per_second(simulation_steps_per_second.max(20).min(60 * 20) as u64);
                    }

                    {
                        if ui.button(im_str!("Reset"), [50.0, DEFAULT_BUTTON_HEIGHT]) {
                            simulation_controller.schedule_restart();
                        }
                        ui.same_line(0.0);
                        if simulation_controller.status == SimulationControllerStatus::Paused {
                            if ui.button(im_str!("Continue  (Space)"), [150.0, DEFAULT_BUTTON_HEIGHT]) {
                                simulation_controller.status = SimulationControllerStatus::Realtime;
                            }
                        } else {
                            if ui.button(im_str!("Pause  (Space)"), [150.0, DEFAULT_BUTTON_HEIGHT]) {
                                simulation_controller.status = SimulationControllerStatus::Paused;
                            }
                        }
                    }
                    {
                        let min_jump = 1.0 / simulation_controller.simulation_steps_per_second() as f32;
                        state.fast_forward_length_seconds = state.fast_forward_length_seconds.max(min_jump);
                        ui.set_next_item_width(50.0);
                        ui.drag_float(im_str!(""), &mut state.fast_forward_length_seconds)
                            .min(min_jump)
                            .max(120.0)
                            .display_format(im_str!("%.2f"))
                            .build();
                        ui.same_line(0.0);
                        if ui.button(im_str!("Fast Forward"), [150.0, DEFAULT_BUTTON_HEIGHT]) {
                            simulation_controller.status = SimulationControllerStatus::FastForward {
                                simulation_jump_length: Duration::from_secs_f32(state.fast_forward_length_seconds),
                            };
                        }
                        ui.same_line(0.0);
                        ui.text_disabled(im_str!("last jump took {:?}", simulation_controller.computation_time_last_fast_forward()));
                    }
                    if let SimulationControllerStatus::Record { .. } = simulation_controller.status {
                        if ui.button(im_str!("End Recording"), [208.0, DEFAULT_BUTTON_HEIGHT]) {
                            simulation_controller.status = SimulationControllerStatus::Paused;
                        }
                    } else {
                        if ui.button(im_str!("Reset & Record Video"), [208.0, DEFAULT_BUTTON_HEIGHT]) {
                            simulation_controller.schedule_restart();
                            for i in 0..usize::MAX {
                                let output_directory = PathBuf::from(format!("recording{}", i));
                                if !output_directory.exists() {
                                    std::fs::create_dir(&output_directory).unwrap();
                                    simulation_controller.status = SimulationControllerStatus::Record { output_directory };
                                    break;
                                }
                            }
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
