use crate::scene::Scene;
use crate::timer::Timer;

// The simulation controller orchestrates simulation steps.
// It holds the central timer and as such is responsible for glueing rendering frames and simulation together.
#[derive(PartialEq, Eq)]
pub enum SimulationControllerStatus {
    Realtime,
    Record,

    Paused,

    FastForward,
}

pub struct SimulationController {
    scheduled_restart: bool,
    pub status: SimulationControllerStatus,
    pub simulation_length: std::time::Duration,
    timer: Timer,
}

// todo: configurable
const SIMULATION_STEP_LENGTH: std::time::Duration = std::time::Duration::from_nanos((1000.0 * 1000.0 * 1000.0 / 120.0) as u64); // 120 simulation steps per second

impl SimulationController {
    pub fn new() -> Self {
        SimulationController {
            scheduled_restart: false,
            status: SimulationControllerStatus::Realtime,
            simulation_length: std::time::Duration::from_secs(60 * 60), // (an hour)
            timer: Timer::new(SIMULATION_STEP_LENGTH),
        }
    }

    pub fn timer(&self) -> &Timer {
        &self.timer
    }

    pub fn on_frame_submitted(&mut self) {
        self.timer.on_frame_submitted();
    }

    pub fn schedule_restart(&mut self) {
        self.scheduled_restart = true;
    }

    pub fn handle_scheduled_restart(&mut self, scene: &mut Scene, device: &wgpu::Device, command_queue: &wgpu::Queue) {
        scene.reset(device, command_queue);
        self.timer = Timer::new(SIMULATION_STEP_LENGTH);
        self.scheduled_restart = false;
        self.status = SimulationControllerStatus::Paused;
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
        self.start_simulation_frame();

        // TODO: Measure time of this.
        while self.status == SimulationControllerStatus::FastForward {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Encoder: Simulation Step Fast Forward"),
            });

            {
                let mut compute_pass = encoder.begin_compute_pass();
                compute_pass.set_bind_group(0, per_frame_bind_group, &[]);

                for _ in 0..MAX_FAST_FORWARD_SIMULATION_BATCH_SIZE {
                    if !self.single_step(scene, &mut compute_pass) {
                        break;
                    }
                }
            }
            queue.submit(&[encoder.finish()]);
            info!("simulation fast forwarding batch submitted");
            device.poll(wgpu::Maintain::Wait); // Seems to be necessary to do the wait every time.
        }

        self.timer.on_frame_submitted();
        self.timer.force_frame_delta(std::time::Duration::from_secs(0));
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
            SimulationControllerStatus::Record => {
                self.timer.force_frame_delta(std::time::Duration::from_secs_f64(1.0 / 60.0));
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
            // 10fps
            std::time::Duration::from_nanos((1000.0 * 1000.0 * 1000.0 / 10.0) as u64)
        } else {
            std::time::Duration::from_secs(999999999999)
        };

        if self.timer.total_simulated_time() + self.timer.simulation_delta() >= self.simulation_length {
            self.status = SimulationControllerStatus::Paused;
            return false;
        }

        if !self.timer.simulation_frame_loop(max_total_step_per_frame) {
            return false;
        }

        scene.step(compute_pass);
        return true;
    }
}
