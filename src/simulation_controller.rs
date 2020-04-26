use crate::scene::Scene;
use crate::timer::{SimulationStepResult, Timer};
use std::time::{Duration, Instant};

// The simulation controller orchestrates simulation steps.
// It holds the central timer and as such is responsible for glueing rendering frames and simulation together.
#[derive(PartialEq, Eq)]
pub enum SimulationControllerStatus {
    Realtime,
    Record {
        output_directory: std::path::PathBuf, // todo: It's weird that the simulation controller knows about the output of a recording.
    },

    Paused,

    FastForward,
}

pub struct SimulationController {
    scheduled_restart: bool,
    timer: Timer,
    computation_time_last_fast_forward: Duration,
    simulation_steps_per_second: u64,
    pub status: SimulationControllerStatus,
    pub simulation_length: Duration,
}

const MIN_REALTIME_FPS: f64 = 10.0;
const RECORDING_FPS: f64 = 60.0;

fn delta_from_steps_per_second(steps_per_second: u64) -> Duration {
    Duration::from_nanos(1000 * 1000 * 1000 / steps_per_second)
}

impl SimulationController {
    pub fn new() -> Self {
        const DEFAULT_SIMULATION_STEPS_PER_SECOND: u64 = 120;

        SimulationController {
            scheduled_restart: false,
            status: SimulationControllerStatus::Realtime,
            simulation_length: Duration::from_secs(60 * 60), // (an hour)
            simulation_steps_per_second: DEFAULT_SIMULATION_STEPS_PER_SECOND,
            timer: Timer::new(delta_from_steps_per_second(DEFAULT_SIMULATION_STEPS_PER_SECOND)),
            computation_time_last_fast_forward: Default::default(),
        }
    }

    pub fn timer(&self) -> &Timer {
        &self.timer
    }

    pub fn on_frame_submitted(&mut self) {
        self.timer.on_frame_submitted();
    }

    pub fn computation_time_last_fast_forward(&self) -> Duration {
        self.computation_time_last_fast_forward
    }

    pub fn schedule_restart(&mut self) {
        self.scheduled_restart = true;
    }

    pub fn simulation_steps_per_second(&self) -> u64 {
        self.simulation_steps_per_second
    }

    pub fn set_simulation_steps_per_second(&mut self, simulation_steps_per_second: u64) {
        self.simulation_steps_per_second = simulation_steps_per_second;
        self.timer
            .set_simulation_delta(delta_from_steps_per_second(self.simulation_steps_per_second));
    }

    pub fn handle_scheduled_restart(&mut self, scene: &mut Scene, device: &wgpu::Device, command_queue: &wgpu::Queue) {
        if !self.scheduled_restart {
            return;
        }

        scene.reset(device, command_queue);
        self.timer = Timer::new(delta_from_steps_per_second(self.simulation_steps_per_second));
        self.scheduled_restart = false;
    }

    // A single fast forward operation is technically just a "very long frame".
    // However, since we need to give the GPU some breathing space it's handled in a different way (-> TDR).
    // Note that we assume that this never happens for realtime & recording, but it well could once a single simulation + render step takes longer than TDR time.
    pub fn fast_forward_steps(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, scene: &Scene, per_frame_bind_group: &wgpu::BindGroup) {
        // TODO: Dynamic estimate to keep batches around 0.5 seconds.
        const MAX_FAST_FORWARD_SIMULATION_BATCH_SIZE: usize = 512;

        if self.status != SimulationControllerStatus::FastForward {
            return;
        }
        let simulation_jump_length = self.simulation_length - self.timer().total_simulated_time();
        self.start_simulation_frame();

        let start_time = Instant::now();
        while self.status == SimulationControllerStatus::FastForward {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Encoder: Simulation Step Fast Forward"),
            });

            let mut batch_size = MAX_FAST_FORWARD_SIMULATION_BATCH_SIZE;
            {
                let mut compute_pass = encoder.begin_compute_pass();
                compute_pass.set_bind_group(0, per_frame_bind_group, &[]);

                for i in 0..MAX_FAST_FORWARD_SIMULATION_BATCH_SIZE {
                    if !self.single_step(scene, &mut compute_pass) {
                        batch_size = i;
                        break;
                    }
                }
            }
            queue.submit(&[encoder.finish()]);
            info!("simulation fast forwarding batch submitted (size {})", batch_size);
            device.poll(wgpu::Maintain::Wait); // Seems to be necessary to do the full wait every time to avoid TDR.
        }
        self.computation_time_last_fast_forward = start_time.elapsed();

        self.timer.on_frame_submitted();
        self.timer.force_frame_delta(Duration::from_secs(0));

        info!(
            "Fast forward of {:?} took {:?} to compute",
            simulation_jump_length, self.computation_time_last_fast_forward
        );
    }

    pub fn frame_steps(&mut self, scene: &Scene, encoder: &mut wgpu::CommandEncoder, per_frame_bind_group: &wgpu::BindGroup) {
        if !self.start_simulation_frame() {
            return;
        }

        let mut compute_pass = encoder.begin_compute_pass();
        compute_pass.set_bind_group(0, per_frame_bind_group, &[]);
        while self.single_step(scene, &mut compute_pass) {}
    }

    fn start_simulation_frame(&mut self) -> bool {
        match self.status {
            SimulationControllerStatus::Realtime => {}
            SimulationControllerStatus::Record { .. } => {
                self.timer.force_frame_delta(Duration::from_secs_f64(1.0 / RECORDING_FPS));
            }
            SimulationControllerStatus::FastForward => {
                self.timer.force_frame_delta(self.simulation_length);
            }
            SimulationControllerStatus::Paused => {
                self.timer.skip_simulation_frame();
                return false;
            }
        };
        return true;
    }

    fn single_step<'a>(&mut self, scene: &'a Scene, compute_pass: &mut wgpu::ComputePass<'a>) -> bool {
        // frame drops are only relevant in realtime mode.
        let max_total_step_per_frame = if self.status == SimulationControllerStatus::Realtime {
            Duration::from_secs_f64(1.0 / MIN_REALTIME_FPS)
        } else {
            Duration::from_secs(u64::MAX)
        };

        if self.timer.total_simulated_time() + self.timer.simulation_delta() >= self.simulation_length {
            self.status = SimulationControllerStatus::Paused;
            return false;
        }

        if self.timer.simulation_frame_loop(max_total_step_per_frame) == SimulationStepResult::PerformStepAndCallAgain {
            scene.step(compute_pass);
            return true;
        }
        return false;
    }
}
