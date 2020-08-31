use crate::renderer::{FluidRenderingMode, SceneRenderer, VolumeVisualizationMode};
use crate::simulation_controller::{SimulationController, SimulationControllerStatus};
use crate::{render_output::screen::Screen, scene::Scene, ApplicationEvent};
use imgui::im_str;
use std::{borrow::Cow, path::PathBuf, time::Duration};
use strum::IntoEnumIterator;
use winit::event_loop::EventLoopProxy;

const SCENE_DIRECTORY: &str = "scenes";

fn list_scene_files() -> Vec<PathBuf> {
    let files: Vec<PathBuf> = std::fs::read_dir(SCENE_DIRECTORY)
        .expect(&format!("Scene directory \"{}\" not present!", SCENE_DIRECTORY))
        .map(|entry| entry.unwrap())
        .filter(|entry| entry.file_type().unwrap().is_file())
        .map(|entry| entry.path())
        .filter(|path| path.extension().unwrap_or_default() == "json")
        .collect();

    if files.len() == 0 {
        panic!("No scene files found scene directory \"{}\"", SCENE_DIRECTORY);
    }

    files
}

pub struct GUIState {
    fast_forward_length_seconds: f32,
    video_fps: i32,
    selected_scene_idx: usize,
    known_scene_files: Vec<PathBuf>,
    wait_for_vblank: bool,
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

        let imgui_renderer = imgui_wgpu::Renderer::new(&mut imgui_context, device, command_queue, Screen::FORMAT_BACKBUFFER, None);

        GUI {
            imgui_context,
            imgui_platform,
            imgui_renderer,

            state: GUIState {
                fast_forward_length_seconds: 5.0,
                video_fps: 60,
                selected_scene_idx: 0,
                known_scene_files: list_scene_files(),
                wait_for_vblank: Screen::DEFAULT_PRESENT_MODE == wgpu::PresentMode::Fifo,
            },
        }
    }

    pub fn selected_scene(&self) -> &PathBuf {
        &self.state.known_scene_files[self.state.selected_scene_idx]
    }

    fn setup_ui(
        ui: &imgui::Ui,
        state: &mut GUIState,
        simulation_controller: &mut SimulationController,
        scene_renderer: &mut SceneRenderer,
        scene: &mut Scene,
        event_loop_proxy: &EventLoopProxy<ApplicationEvent>,
    ) {
        const DEFAULT_BUTTON_HEIGHT: f32 = 19.0;

        let window = imgui::Window::new(im_str!("Blub"));
        window
            .position([0.0, 0.0], imgui::Condition::FirstUseEver)
            .always_auto_resize(true)
            .build(&ui, || {
                //////////////////////////////////////////////////
                // Timing information
                //////////////////////////////////////////////////
                {
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
                    if ui.checkbox(im_str!("wait for vsync"), &mut state.wait_for_vblank) {
                        let present_mode = match state.wait_for_vblank {
                            true => wgpu::PresentMode::Fifo,
                            false => wgpu::PresentMode::Mailbox,
                        };
                        event_loop_proxy.send_event(ApplicationEvent::ChangePresentMode(present_mode)).unwrap();
                    }
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
                }
                if imgui::CollapsingHeader::new(im_str!("Solver")).build(&ui) {
                    let fluid = scene.fluid_mut();

                    let stack_token = ui.push_id(1);
                    {
                        let stats = fluid.pressure_solver_stats_velocity();
                        ui.text(im_str!("pressure solver, primary (from velocity)"));
                        ui.text(im_str!("mse: {:.4}", stats.last_mse));
                        ui.text(im_str!("# solver iterations: {}", stats.last_iteration_count));

                        let config = fluid.pressure_solver_config_velocity();
                        ui.drag_float(im_str!("target mse"), &mut config.target_mse)
                            .min(0.0001)
                            .max(1.0)
                            .speed(0.0001)
                            .display_format(im_str!("%.4f"))
                            .build();
                        let mut min_iteration_count = config.min_num_iterations as i32;
                        let mut max_iteration_count = config.max_num_iterations as i32;
                        if ui
                            .drag_int_range2(im_str!("min/max iteration count"), &mut min_iteration_count, &mut max_iteration_count)
                            .min(2)
                            .max(100)
                            .build()
                        {
                            config.min_num_iterations = min_iteration_count as u32;
                            config.max_num_iterations = max_iteration_count as u32;
                        }
                    }
                    stack_token.pop(ui);
                    ui.separator();
                    {
                        let stats = fluid.pressure_solver_stats_density();
                        ui.text(im_str!("pressure solver, secondary (from density)"));
                        ui.text(im_str!("mse: {:.4}", stats.last_mse));
                        ui.text(im_str!("# solver iterations: {}", stats.last_iteration_count));

                        let config = fluid.pressure_solver_config_density();
                        ui.drag_float(im_str!("target mse"), &mut config.target_mse)
                            .min(0.0001)
                            .max(1.0)
                            .speed(0.0001)
                            .display_format(im_str!("%.4f"))
                            .build();
                        let mut min_iteration_count = config.min_num_iterations as i32;
                        let mut max_iteration_count = config.max_num_iterations as i32;
                        if ui
                            .drag_int_range2(im_str!("min/max iteration count"), &mut min_iteration_count, &mut max_iteration_count)
                            .min(2)
                            .max(100)
                            .build()
                        {
                            config.min_num_iterations = min_iteration_count as u32;
                            config.max_num_iterations = max_iteration_count as u32;
                        }
                    }
                }

                if imgui::CollapsingHeader::new(im_str!("Simulation Controller & Recording")).build(&ui) {
                    ui.text(im_str!(
                        "total num simulation steps: {}",
                        simulation_controller.timer().num_simulation_steps_performed()
                    ));
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

                    if ui
                        .input_float(im_str!("time scale"), &mut simulation_controller.time_scale)
                        .step(0.05)
                        .enter_returns_true(true)
                        .build()
                    {
                        simulation_controller.time_scale = simulation_controller.time_scale.max(0.01).min(100.0);
                    }

                    {
                        if ui.button(im_str!("Reset"), [50.0, DEFAULT_BUTTON_HEIGHT]) {
                            event_loop_proxy.send_event(ApplicationEvent::ResetScene).unwrap();
                        }
                        ui.same_line(0.0);
                        if ui.button(
                            if simulation_controller.status() == SimulationControllerStatus::Paused {
                                im_str!("Continue  (Space)")
                            } else {
                                im_str!("Pause  (Space)")
                            },
                            [150.0, DEFAULT_BUTTON_HEIGHT],
                        ) {
                            simulation_controller.resume_realtime();
                        }
                    }
                    {
                        let min_jump = 1.0 / simulation_controller.simulation_steps_per_second() as f32;
                        state.fast_forward_length_seconds = state.fast_forward_length_seconds.max(min_jump);
                        ui.set_next_item_width(50.0);
                        ui.drag_float(im_str!(""), &mut state.fast_forward_length_seconds)
                            .min(min_jump)
                            .max(120.0)
                            .speed(0.005)
                            .display_format(im_str!("%.2f"))
                            .build();
                        ui.same_line(0.0);
                        if ui.button(im_str!("Fast Forward"), [150.0, DEFAULT_BUTTON_HEIGHT]) {
                            event_loop_proxy
                                .send_event(ApplicationEvent::FastForwardSimulation(Duration::from_secs_f32(
                                    state.fast_forward_length_seconds,
                                )))
                                .unwrap();
                        }
                        ui.same_line(0.0);
                        ui.text_disabled(im_str!("last jump took {:?}", simulation_controller.computation_time_last_fast_forward()));
                    }

                    if let SimulationControllerStatus::RecordingWithFixedFrameLength { .. } = simulation_controller.status() {
                        if ui.button(im_str!("End Recording"), [208.0, DEFAULT_BUTTON_HEIGHT]) {
                            simulation_controller.pause_or_resume();
                        }
                    } else {
                        if ui.button(im_str!("Reset & Record Video"), [208.0, DEFAULT_BUTTON_HEIGHT]) {
                            event_loop_proxy
                                .send_event(ApplicationEvent::ResetAndStartRecording {
                                    recording_fps: state.video_fps as f64,
                                })
                                .unwrap();
                        }
                        ui.same_line(0.0);
                        ui.set_next_item_width(40.0);
                        ui.drag_int(im_str!("video fps"), &mut state.video_fps).min(10).max(300).build();
                    }
                }

                if imgui::CollapsingHeader::new(im_str!("Scene Settings")).build(&ui) {
                    ui.set_next_item_width(150.0);
                    if imgui::ComboBox::new(im_str!("Load Scene")).build_simple(
                        ui,
                        &mut state.selected_scene_idx,
                        &state.known_scene_files,
                        &|path| Cow::from(im_str!("{:?}", path.strip_prefix(SCENE_DIRECTORY).unwrap_or(path))),
                    ) {
                        event_loop_proxy
                            .send_event(ApplicationEvent::LoadScene(state.known_scene_files[state.selected_scene_idx].clone()))
                            .unwrap();
                    }
                }

                if imgui::CollapsingHeader::new(im_str!("Rendering Settings")).build(&ui) {
                    {
                        let mut current_fluid_rendering = scene_renderer.fluid_rendering_mode as usize;
                        imgui::ComboBox::new(im_str!("Fluid Rendering")).build_simple(
                            ui,
                            &mut current_fluid_rendering,
                            &FluidRenderingMode::iter().collect::<Vec<FluidRenderingMode>>(),
                            &|value| Cow::from(im_str!("{:?}", *value)),
                        );
                        scene_renderer.fluid_rendering_mode = FluidRenderingMode::iter().skip(current_fluid_rendering).next().unwrap();
                    }
                    {
                        ui.drag_float(im_str!("Particle Radius Factor"), &mut scene_renderer.particle_radius_factor)
                            .min(0.0)
                            .max(1.0)
                            .speed(0.01)
                            .display_format(im_str!("%.2f"))
                            .build();
                    }
                    {
                        let mut current_volume_visualization = scene_renderer.volume_visualization as usize;
                        imgui::ComboBox::new(im_str!("Volume Visualization")).build_simple(
                            ui,
                            &mut current_volume_visualization,
                            &VolumeVisualizationMode::iter().collect::<Vec<VolumeVisualizationMode>>(),
                            &|value| Cow::from(im_str!("{:?}", *value)),
                        );
                        scene_renderer.volume_visualization = VolumeVisualizationMode::iter().skip(current_volume_visualization).next().unwrap();
                    }
                    ui.drag_float(im_str!("Velocity Visualization Scale"), &mut scene_renderer.velocity_visualization_scale)
                        .min(0.001)
                        .max(5.0)
                        .speed(0.0001)
                        .display_format(im_str!("%.3f"))
                        .build();
                    ui.checkbox(im_str!("Show Fluid Domain Bounds"), &mut scene_renderer.enable_box_lines);
                }
            });
    }

    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        window: &winit::window::Window,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        simulation_controller: &mut SimulationController,
        scene_renderer: &mut SceneRenderer,
        scene: &mut Scene,
        event_loop_proxy: &EventLoopProxy<ApplicationEvent>,
    ) {
        let context = &mut self.imgui_context;
        let state = &mut self.state;

        self.imgui_platform
            .prepare_frame(context.io_mut(), window)
            .expect("Failed to prepare imgui frame");
        let ui = context.frame();
        Self::setup_ui(&ui, state, simulation_controller, scene_renderer, scene, event_loop_proxy);
        self.imgui_platform.prepare_render(&ui, &window);
        self.imgui_renderer
            .render(ui.render(), &device, encoder, queue, view)
            .expect("IMGUI rendering failed");
    }

    pub fn handle_event<T>(&mut self, window: &winit::window::Window, event: &winit::event::Event<T>) {
        self.imgui_platform.handle_event(self.imgui_context.io_mut(), window, event);
    }
}
