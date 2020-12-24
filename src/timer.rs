use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

// Timer keeps track of render & simulation timing and time statistics.
// It's set up in a way that makes "normal realtime rendering" easy, but has hooks to allow special handling
// (simulation jump, simulation pause, recording)
//
// There is three dependent clocks:
// * real time
//      that's the watch on your wrist
// * render time
//      same as on your watch if you're not recording or fast forwarding to a specific time
// * simulation time
//      tries to keep up with render time but in different chunks and may start to drop steps
//
// Note that since our simulation is all on GPU it doesn't make sense to take timings around simulation steps on CPU!
pub struct Timer {
    // real time measures
    timestamp_last_frame: Instant,
    duration_last_frame: Duration,
    frame_duration_history: VecDeque<Duration>,

    // render time measures
    total_rendered_time: Duration,
    current_frame_delta: Duration,
    num_frames_rendered: u32,

    // simulation time
    simulation_delta: Duration,
    num_simulation_steps: u32,
    num_simulation_steps_this_frame: u32,
    total_simulated_time: Duration,
    accepted_simulation_to_render_lag: Duration, // time lost that we don't plan on catching up anymore
}

#[derive(PartialEq, Eq)]
pub enum SimulationStepResult {
    PerformStepAndCallAgain,

    CaughtUpWithRenderTime,
    DroppingSimulationSteps,
}

const FRAME_DURATION_HISTORY_LENGTH: usize = 50;

impl Timer {
    pub fn new(simulation_delta: Duration) -> Timer {
        Timer {
            timestamp_last_frame: Instant::now(),
            duration_last_frame: Duration::from_millis(0),
            frame_duration_history: VecDeque::with_capacity(FRAME_DURATION_HISTORY_LENGTH),

            total_rendered_time: Duration::from_millis(0),
            current_frame_delta: Duration::from_millis(0),
            num_frames_rendered: 0,

            simulation_delta,
            num_simulation_steps: 0,
            num_simulation_steps_this_frame: 0,
            total_simulated_time: Duration::from_millis(0),
            accepted_simulation_to_render_lag: Duration::from_millis(0),
        }
    }

    // Forces a given frame delta (timestep on the rendering timeline)
    // Usually the frame delta is just the time between the last two on_frame_submitted calls, but this overwrites it.
    // Useful to jump to a specific time (recording, or fast forwarding the simulation).
    pub fn force_frame_delta(&mut self, delta: Duration) {
        self.total_rendered_time -= self.current_frame_delta;
        self.current_frame_delta = delta;
        self.total_rendered_time += self.current_frame_delta;
    }

    pub fn on_frame_submitted(&mut self, time_scale: f32) {
        self.duration_last_frame = self.timestamp_last_frame.elapsed();
        if self.frame_duration_history.len() == FRAME_DURATION_HISTORY_LENGTH {
            self.frame_duration_history.pop_front();
        }
        self.frame_duration_history.push_back(self.duration_last_frame);
        self.current_frame_delta = self.duration_last_frame.mul_f32(time_scale);
        self.total_rendered_time += self.current_frame_delta;

        self.timestamp_last_frame = std::time::Instant::now();
        self.num_simulation_steps_this_frame = 0;
        self.num_frames_rendered += 1;
    }

    pub fn skip_simulation_frame(&mut self) {
        self.accepted_simulation_to_render_lag += self.current_frame_delta;
    }

    pub fn simulation_frame_loop(&mut self, max_total_step_per_frame: Duration) -> SimulationStepResult {
        // simulation time shouldn't advance faster than render time
        let residual_time = self
            .total_rendered_time
            .checked_sub(self.total_simulated_time + self.accepted_simulation_to_render_lag)
            .unwrap();
        if residual_time < self.simulation_delta {
            // println!(
            //     "realtime {}, fps {}",
            //     self.num_simulation_steps_this_frame,
            //     1.0 / self.frame_delta().as_secs_f32()
            // );
            return SimulationStepResult::CaughtUpWithRenderTime;
        }

        // Did we hit a maximum of simulation steps and want to introduce lag instead?
        if self.num_simulation_steps_this_frame * self.simulation_delta > max_total_step_per_frame {
            // We heuristically don't drop all lost simulation frames. This avoids oscillating between realtime and offline
            // which is caused by our frame deltas being influenced by work from a couple of cpu frames ago (due gpu/cpu sync)
            self.accepted_simulation_to_render_lag += residual_time.mul_f32(0.9);
            // println!(
            //     "lagtime {}, fps {}",
            //     self.num_simulation_steps_this_frame,
            //     1.0 / self.frame_delta().as_secs_f32()
            // );
            return SimulationStepResult::DroppingSimulationSteps;
        }

        self.num_simulation_steps_this_frame += 1;
        self.num_simulation_steps += 1;
        self.total_simulated_time += self.simulation_delta;
        SimulationStepResult::PerformStepAndCallAgain
    }

    pub fn simulation_delta(&self) -> Duration {
        self.simulation_delta
    }

    pub fn set_simulation_delta(&mut self, delta: Duration) {
        self.simulation_delta = delta;
    }

    pub fn frame_delta(&self) -> Duration {
        self.current_frame_delta
    }

    // Duration of the previous frame. (this is not necessarily equal to the frame time delta!)
    pub fn duration_last_frame(&self) -> Duration {
        self.duration_last_frame
    }

    pub fn duration_last_frame_history(&self) -> &VecDeque<Duration> {
        &self.frame_duration_history
    }

    // Total render time, including current frame. (equal to real time if not configured otherwise!)
    pub fn total_render_time(&self) -> Duration {
        self.total_rendered_time
    }
    // Total time simulated
    pub fn total_simulated_time(&self) -> Duration {
        self.total_simulated_time
    }

    pub fn num_simulation_steps_performed_for_current_frame(&self) -> u32 {
        self.num_simulation_steps_this_frame
    }

    pub fn num_simulation_steps_performed(&self) -> u32 {
        self.num_simulation_steps
    }

    pub fn fill_global_uniform_buffer(&self) -> FrameTimeUniformBufferContent {
        FrameTimeUniformBufferContent {
            total_passed: self.total_rendered_time.as_secs_f32(),
            frame_delta: self.current_frame_delta.as_secs_f32(),
            total_simulated_time: self.total_simulated_time.as_secs_f32(),
            simulation_delta: self.simulation_delta.as_secs_f32(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct FrameTimeUniformBufferContent {
    pub total_passed: f32,         // How much time has passed on the rendering clock since rendering started.
    pub frame_delta: f32,          // How long a previous frame took in seconds.
    pub total_simulated_time: f32, // How much time has passed in the simulation *excluding any steps in the current frame*
    pub simulation_delta: f32,     // How much we're advancing the simulation for each step in the current frame.
}
