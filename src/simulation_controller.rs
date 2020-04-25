use crate::hybrid_fluid::HybridFluid;
use crate::timer::Timer;

#[derive(PartialEq, Eq)]
pub enum SimulationControllerStatus {
    Realtime,
    Record,

    Paused,

    FastForward,
}

pub struct SimulationController {
    pub scheduled_restart: bool,
    pub status: SimulationControllerStatus,
    pub simulation_length: std::time::Duration,
    pub timer: Timer,
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

    pub fn restart(&mut self, hybrid_fluid: &mut HybridFluid, device: &wgpu::Device, command_queue: &wgpu::Queue) {
        let mut init_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Particle Init Encoder"),
        });

        hybrid_fluid.reset();
        hybrid_fluid.add_fluid_cube(
            device,
            &mut init_encoder,
            cgmath::Point3::new(0.0, 0.0, 0.0),
            cgmath::Point3::new(64.0, 40.0, 64.0),
        );

        command_queue.submit(&[init_encoder.finish()]);
        device.poll(wgpu::Maintain::Wait);

        self.timer = Timer::new(SIMULATION_STEP_LENGTH);

        self.scheduled_restart = false;
        self.status = SimulationControllerStatus::Paused;
    }

    // A single fast forward operation is technically just a "very long frame".
    // However, since we need to give the GPU some breathing space it's handled in a different way (-> TDR).
    // Note that we assume that this never happens for realtime & recording, but it well could once a single simulation + render step takes longer than TDR time.
    pub fn do_fast_forward_steps(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        hybrid_fluid: &HybridFluid,
        per_frame_bind_group: &wgpu::BindGroup,
    ) {
        const MAX_FAST_FORWARD_SIMULATION_BATCH_SIZE: usize = 64;

        if self.status != SimulationControllerStatus::FastForward {
            return;
        }

        // TODO: Measure time of this.

        while self.status == SimulationControllerStatus::FastForward {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Encoder: Simulation Step Fast Forward"),
            });

            {
                let mut compute_pass = encoder.begin_compute_pass();
                compute_pass.set_bind_group(0, per_frame_bind_group, &[]);

                for _ in 0..MAX_FAST_FORWARD_SIMULATION_BATCH_SIZE {
                    if !self.single_step(hybrid_fluid, &mut compute_pass) {
                        break;
                    }
                }
            }
            queue.submit(&[encoder.finish()]);
            info!("simulation fast forwarding batch submitted");
            device.poll(wgpu::Maintain::Wait); // TODO?
        }

        self.timer.on_frame_submitted();
        self.timer.force_frame_delta(std::time::Duration::from_secs(0));
    }

    pub fn do_frame_steps(&mut self, hybrid_fluid: &HybridFluid, encoder: &mut wgpu::CommandEncoder, per_frame_bind_group: &wgpu::BindGroup) {
        let mut compute_pass = encoder.begin_compute_pass();
        compute_pass.set_bind_group(0, per_frame_bind_group, &[]);

        while self.single_step(hybrid_fluid, &mut compute_pass) {}
    }

    fn single_step<'a>(&mut self, hybrid_fluid: &'a HybridFluid, compute_pass: &mut wgpu::ComputePass<'a>) -> bool {
        // TODO: Move some of this time manipulation into the timer

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

        // frame drops are only relevant in realtime mode.
        let max_total_step_per_frame = if self.status == SimulationControllerStatus::Realtime {
            std::time::Duration::from_nanos((1000.0 * 1000.0 * 1000.0 / 10.0) as u64)
        } else {
            std::time::Duration::from_secs(999999999999)
        };

        if self.timer.total_simulated_time() + self.timer.simulation_delta() >= self.simulation_length {
            self.status = SimulationControllerStatus::Paused;
            return false;
        }

        if !self.timer.simulation_step_loop(max_total_step_per_frame) {
            return false;
        }

        hybrid_fluid.step(compute_pass);
        return true;
    }
}
