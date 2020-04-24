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
        self.status = SimulationControllerStatus::Realtime;
    }

    pub fn simulate_step(&mut self, hybrid_fluid: &HybridFluid, encoder: &mut wgpu::CommandEncoder, per_frame_bind_group: &wgpu::BindGroup) {
        // maximum number of steps per frame
        let max_total_step_per_frame = match self.status {
            SimulationControllerStatus::Realtime => {
                // stop catching up if slower than at 10fps
                std::time::Duration::from_nanos((1000.0 * 1000.0 * 1000.0 / 10.0) as u64)
            }
            SimulationControllerStatus::Record => {
                // todo, shouldn't be hardcoded
                let delta = std::time::Duration::from_secs_f64(1.0 / 60.0);
                self.timer.force_frame_delta(delta);
                delta
            }
            SimulationControllerStatus::FastForward => {
                self.timer.force_frame_delta(self.simulation_length);
                self.simulation_length
            }
            SimulationControllerStatus::Paused => {
                self.timer.skip_simulation_frame();
                return;
            }
        };

        let mut cpass = encoder.begin_compute_pass();
        cpass.set_bind_group(0, per_frame_bind_group, &[]);

        loop {
            if self.timer.total_simulated_time() >= self.simulation_length {
                self.status = SimulationControllerStatus::Paused;
                return;
            }

            if !self.timer.simulation_step_loop(max_total_step_per_frame) {
                return;
            }

            hybrid_fluid.step(&mut cpass);
        }
    }
}
