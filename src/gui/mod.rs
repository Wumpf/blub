use crate::renderer::{FluidRenderingMode, SceneRenderer, VolumeVisualizationMode};
use crate::simulation_controller::{SimulationController, SimulationControllerStatus};
use crate::{
    render_output::screen::Screen,
    scene::Scene,
    simulation::{HybridFluid, SolverConfig, SolverStatisticSample},
    ApplicationEvent,
};
use std::{collections::VecDeque, path::PathBuf, time::Duration};
use strum::IntoEnumIterator;
use winit::event_loop::EventLoopProxy;

mod custom_widgets;

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
    platform: egui_winit_platform::Platform,
    render_pass: egui_wgpu_backend::RenderPass,

    state: GUIState,
}

struct DummyRepaintSignal;
impl epi::RepaintSignal for DummyRepaintSignal {
    fn request_repaint(&self) {}
}

impl GUI {
    pub fn new(device: &wgpu::Device, window: &winit::window::Window) -> Self {
        let platform = egui_winit_platform::Platform::new(egui_winit_platform::PlatformDescriptor {
            physical_width: window.inner_size().width as u32,
            physical_height: window.inner_size().height as u32,
            scale_factor: window.scale_factor(),
            font_definitions: egui::FontDefinitions::default(),
            style: Default::default(),
        });

        let render_pass = egui_wgpu_backend::RenderPass::new(device, Screen::FORMAT_BACKBUFFER);

        GUI {
            platform,
            render_pass,
            state: GUIState {
                fast_forward_length_seconds: 5.0,
                video_fps: 60,
                selected_scene_idx: 0,
                known_scene_files: list_scene_files(),
                wait_for_vblank: Screen::DEFAULT_PRESENT_MODE == wgpu::PresentMode::Fifo,
            },
        }
    }

    pub fn handle_event<T>(&mut self, winit_event: &winit::event::Event<T>) {
        self.platform.handle_event(winit_event);
    }

    pub fn selected_scene(&self) -> &PathBuf {
        &self.state.known_scene_files[self.state.selected_scene_idx]
    }

    fn setup_ui_timer(
        ui: &mut egui::Ui,
        state: &mut GUIState,
        simulation_controller: &SimulationController,
        event_loop_proxy: &EventLoopProxy<ApplicationEvent>,
    ) {
        ui.add(
            egui::Label::new(format!(
                "{:3.2}ms, FPS: {:3.2}",
                simulation_controller.timer().duration_last_frame().as_secs_f64() * 1000.0,
                1000.0 / 1000.0 / simulation_controller.timer().duration_last_frame().as_secs_f64()
            ))
            .heading(),
        );

        let frame_times = simulation_controller
            .timer()
            .duration_last_frame_history()
            .iter()
            .map(|d| d.as_secs_f32() * 1000.0)
            .collect::<Vec<f32>>();
        custom_widgets::plot_histogram(ui, 40.0, &frame_times, frame_times.iter().cloned().fold(0.0, f32::max), "ms");

        if ui.checkbox(&mut state.wait_for_vblank, "wait for vsync").clicked() {
            let present_mode = match state.wait_for_vblank {
                true => wgpu::PresentMode::Fifo,
                false => wgpu::PresentMode::Mailbox,
            };
            event_loop_proxy.send_event(ApplicationEvent::ChangePresentMode(present_mode)).unwrap();
        }
        ui.separator();
        ui.label(format!(
            "num simulation steps current frame: {}",
            simulation_controller.timer().num_simulation_steps_performed_for_current_frame()
        ));
        if let SimulationControllerStatus::RecordingWithFixedFrameLength { .. } = simulation_controller.status() {
            ui.add(
                egui::Label::new(format!(
                    "OFFLINE RECORDING - rendered time forced to {:.2}fps",
                    1.0 / simulation_controller.timer().frame_delta().as_secs_f64()
                ))
                .text_color(egui::Color32::RED),
            );
        } else {
            ui.label(format!(
                "rendered time:  {:.2}",
                simulation_controller.timer().total_render_time().as_secs_f64()
            ));
        }
        ui.label(format!(
            "simulated time: {:.2}",
            simulation_controller.timer().total_simulated_time().as_secs_f64()
        ));
    }

    fn setup_ui_solver_stats(ui: &mut egui::Ui, stats: &VecDeque<SolverStatisticSample>, max_iterations: i32, error_tolerance: f32) {
        let newest_sample = match stats.back() {
            Some(&sample) => sample,
            None => Default::default(),
        };
        custom_widgets::plot_histogram(
            ui,
            40.0,
            &stats.iter().map(|sample| sample.error).collect::<Vec<f32>>(),
            error_tolerance * 3.0,
            "",
        );
        ui.label(&format!("max residual error - {}", newest_sample.error));

        custom_widgets::plot_histogram(
            ui,
            40.0,
            &stats.iter().map(|sample| sample.iteration_count as f32).collect::<Vec<f32>>(),
            max_iterations as f32,
            "",
        );
        ui.label(&format!("# solver iterations - {}", newest_sample.iteration_count));
    }

    fn setup_ui_solver_config(ui: &mut egui::Ui, config: &mut SolverConfig) {
        ui.add(egui::Slider::f32(&mut config.error_tolerance, 0.0001..=1.0).text("error tolerance"));
        ui.add(egui::Slider::i32(&mut config.max_num_iterations, 2..=128).text("max iteration count"));
        ui.add(egui::Slider::i32(&mut config.error_check_frequency, 1..=config.max_num_iterations).text("error check frequency count"));
    }

    fn setup_ui_solver(ui: &mut egui::Ui, fluid: &mut HybridFluid) {
        {
            ui.label("pressure solver, primary (from velocity)");
            let max_num_iterations = fluid.pressure_solver_config_velocity().max_num_iterations;
            let error_tolerance = fluid.pressure_solver_config_velocity().error_tolerance;
            Self::setup_ui_solver_stats(ui, fluid.pressure_solver_stats_velocity(), max_num_iterations, error_tolerance);
            //Self::setup_ui_solver_config(ui, fluid.pressure_solver_config_velocity());
        }
        ui.separator();
        {
            ui.label("pressure solver, secondary (from density)");
            let max_num_iterations = fluid.pressure_solver_config_density().max_num_iterations;
            let error_tolerance = fluid.pressure_solver_config_density().error_tolerance;
            Self::setup_ui_solver_stats(ui, fluid.pressure_solver_stats_density(), max_num_iterations, error_tolerance);
            //Self::setup_ui_solver_config(ui, fluid.pressure_solver_config_density());
        }
        // One config for both
        ui.separator();
        {
            Self::setup_ui_solver_config(ui, fluid.pressure_solver_config_density());
            *fluid.pressure_solver_config_velocity() = *fluid.pressure_solver_config_density()
        }
    }

    fn setup_ui_simulation_control(
        ui: &mut egui::Ui,
        state: &mut GUIState,
        simulation_controller: &mut SimulationController,
        event_loop_proxy: &EventLoopProxy<ApplicationEvent>,
    ) {
        ui.label(format!(
            "total num simulation steps: {}",
            simulation_controller.timer().num_simulation_steps_performed()
        ));

        ui.horizontal(|ui| {
            let mut simulation_time_seconds = simulation_controller.simulation_stop_time.as_secs_f32();
            ui.add(egui::DragValue::f32(&mut simulation_time_seconds).speed(0.1));
            simulation_controller.simulation_stop_time = std::time::Duration::from_secs_f32(simulation_time_seconds);
            ui.label("target simulation time (s)");
        });

        ui.horizontal(|ui| {
            let mut simulation_steps_per_second = simulation_controller.simulation_steps_per_second() as i32;
            ui.add(egui::DragValue::i32(&mut simulation_steps_per_second).speed(10.0));
            simulation_controller.set_simulation_steps_per_second(simulation_steps_per_second.max(20).min(60 * 20) as u64);
            ui.label("simulation steps per second");
        });

        ui.horizontal(|ui| {
            ui.add(
                egui::DragValue::f32(&mut simulation_controller.time_scale)
                    .speed(0.05)
                    .clamp_range(0.01..=100.0),
            );
            ui.label("time scale");
        });

        ui.horizontal(|ui| {
            if ui.button("Reset").clicked() {
                event_loop_proxy.send_event(ApplicationEvent::ResetScene).unwrap();
            }
            if ui
                .button(if simulation_controller.status() == SimulationControllerStatus::Paused {
                    "Continue  (Space)"
                } else {
                    "Pause  (Space)"
                })
                .clicked()
            {
                simulation_controller.pause_or_resume();
            }
        });

        ui.horizontal(|ui| {
            let min_jump = 1.0 / simulation_controller.simulation_steps_per_second() as f32;
            state.fast_forward_length_seconds = state.fast_forward_length_seconds.max(min_jump);
            ui.add(
                egui::DragValue::f32(&mut state.fast_forward_length_seconds)
                    .speed(0.01)
                    .clamp_range(min_jump..=120.0),
            );
            if ui.button("Fast Forward").clicked() {
                event_loop_proxy
                    .send_event(ApplicationEvent::FastForwardSimulation(Duration::from_secs_f32(
                        state.fast_forward_length_seconds,
                    )))
                    .unwrap();
            }
            ui.label(format!("last jump took {:?}", simulation_controller.computation_time_last_fast_forward()));
        });

        if let SimulationControllerStatus::RecordingWithFixedFrameLength { .. } = simulation_controller.status() {
            if ui.button("End Recording").clicked() {
                simulation_controller.pause_or_resume();
            }
        } else {
            ui.horizontal(|ui| {
                if ui.button("Reset & Record Video").clicked() {
                    event_loop_proxy
                        .send_event(ApplicationEvent::ResetAndStartRecording {
                            recording_fps: state.video_fps as f64,
                        })
                        .unwrap();
                }

                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::i32(&mut state.video_fps).clamp_range(10.0..=300.0));
                    ui.label("video fps")
                });
            });
        }
    }

    fn setup_ui_rendersettings(ui: &mut egui::Ui, scene_renderer: &mut SceneRenderer) {
        egui::combo_box_with_label(ui, "Fluid Rendering", format!("{:?}", scene_renderer.fluid_rendering_mode), |ui| {
            for mode in FluidRenderingMode::iter() {
                ui.selectable_value(&mut scene_renderer.fluid_rendering_mode, mode, format!("{:?}", mode));
            }
        });
        ui.add(egui::Slider::f32(&mut scene_renderer.particle_radius_factor, 0.0..=1.0).text("Particle Radius Factor"));
        egui::combo_box_with_label(ui, "Volume Visualization", format!("{:?}", scene_renderer.volume_visualization), |ui| {
            for mode in VolumeVisualizationMode::iter() {
                ui.selectable_value(&mut scene_renderer.volume_visualization, mode, format!("{:?}", mode));
            }
        });
        ui.add(
            egui::Slider::f32(&mut scene_renderer.velocity_visualization_scale, 0.001..=5.0)
                .logarithmic(true)
                .text("Velocity Visualization Scale"),
        );
        ui.checkbox(&mut scene_renderer.enable_mesh_rendering, "Render meshes");
        ui.checkbox(&mut scene_renderer.enable_box_lines, "Show Fluid Domain Bounds");
    }

    pub fn draw(
        &mut self,
        device: &mut wgpu::Device,
        window: &winit::window::Window,
        encoder: &mut wgpu::CommandEncoder,
        queue: &mut wgpu::Queue,
        view: &wgpu::TextureView,
        simulation_controller: &mut SimulationController,
        scene_renderer: &mut SceneRenderer,
        scene: &mut Scene,
        event_loop_proxy: &EventLoopProxy<ApplicationEvent>,
    ) {
        self.platform.begin_frame();

        // Draw gui
        egui::Window::new("Blub").show(&self.platform.context(), |ui| {
            Self::setup_ui_timer(ui, &mut self.state, simulation_controller, event_loop_proxy);

            egui::CollapsingHeader::new("Solver").show(ui, |ui| {
                Self::setup_ui_solver(ui, scene.fluid_mut());
            });
            egui::CollapsingHeader::new("Simulation Controller & Recording")
                .default_open(true)
                .show(ui, |ui| {
                    Self::setup_ui_simulation_control(ui, &mut self.state, simulation_controller, event_loop_proxy);
                });
            egui::CollapsingHeader::new("Rendering Settings").default_open(true).show(ui, |ui| {
                Self::setup_ui_rendersettings(ui, scene_renderer);
            });
        });

        // End the UI frame.
        let (_output, paint_commands) = self.platform.end_frame();
        let paint_jobs = self.platform.context().tessellate(paint_commands);

        // Upload all resources for the GPU.
        let screen_descriptor = egui_wgpu_backend::ScreenDescriptor {
            physical_width: window.inner_size().width,
            physical_height: window.inner_size().height,
            scale_factor: window.scale_factor() as f32,
        };
        self.render_pass.update_texture(device, queue, &self.platform.context().texture());
        self.render_pass.update_user_textures(device, queue);
        self.render_pass.update_buffers(device, queue, &paint_jobs, &screen_descriptor);

        // Record all render passes.
        self.render_pass.execute(encoder, view, &paint_jobs, &screen_descriptor, None);
    }
}