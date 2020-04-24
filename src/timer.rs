use std::time::{Duration, Instant};

#[allow(dead_code)]
pub enum TimeConfiguration {
    // Every frame has a fixed number of steps. (simulation step length varies with frame time!)
    RealtimeRenderingFixedSimulationStepCountPerFrame(u32),
    // Given fixed timestep.
    RealtimeRenderingFixedSimulationStep { simulation_delta: Duration },
}

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
    config: TimeConfiguration,

    // real time measures
    timestamp_last_frame: Instant,
    time_since_last_frame_submitted: Duration,

    // todo: Keep statistics over the last couple of render frames for display of smooth timings

    // render time measures
    total_rendered_time: Duration,
    current_frame_delta: Duration,

    // simulation time
    num_simulation_steps: u32,
    num_simulation_steps_this_frame: u32,
    total_simulated_time: Duration,
    simulation_current_frame_passed: Duration,
    accepted_simulation_to_render_lag: Duration, // time lost that we don't plan on catching up anymore
}

impl Timer {
    pub fn new(config: TimeConfiguration) -> Timer {
        Timer {
            config,

            timestamp_last_frame: Instant::now(),
            time_since_last_frame_submitted: Duration::from_millis(0),

            total_rendered_time: Duration::from_millis(0),
            current_frame_delta: Duration::from_millis(0),

            num_simulation_steps: 0,
            num_simulation_steps_this_frame: 0,
            total_simulated_time: Duration::from_millis(0),
            simulation_current_frame_passed: Duration::from_millis(0),
            accepted_simulation_to_render_lag: Duration::from_millis(0),
        }
    }

    // Forces a given frame delta (timestep on the rendering timeline)
    // Usually the frame delta is just the time between the last two on_frame_submitted calls, but this overwrites this.
    // Useful to jump to a specific time (recording, or fast forwarding the simulation).
    pub fn force_frame_delta(&mut self, delta: Duration) {
        self.current_frame_delta = delta;
    }

    pub fn on_frame_submitted(&mut self) {
        // Frame is submitted, so we finally can advance the render time.
        self.total_rendered_time += self.current_frame_delta;

        self.time_since_last_frame_submitted = self.timestamp_last_frame.elapsed();
        self.current_frame_delta = self.time_since_last_frame_submitted;

        self.timestamp_last_frame = std::time::Instant::now();
        self.simulation_current_frame_passed = Duration::from_millis(0);
        self.num_simulation_steps_this_frame = 0;
    }

    pub fn skip_simulation_frame(&mut self) {
        self.accepted_simulation_to_render_lag += self.current_frame_delta;
    }

    pub fn simulation_step_loop(&mut self, max_total_step_per_frame: Duration) -> bool {
        let simulation_delta = self.simulation_delta();
        if self.num_simulation_steps_this_frame > 0 {
            self.total_simulated_time += simulation_delta;
            self.simulation_current_frame_passed += simulation_delta;
        }

        // simulation time shouldn't advance faster than render time
        let residual_time = (self.total_rendered_time + self.current_frame_delta)
            .checked_sub(self.total_simulated_time + self.accepted_simulation_to_render_lag)
            .unwrap();
        if residual_time < simulation_delta {
            // println!(
            //     "realtime {}, fps {}",
            //     self.num_simulation_steps_this_frame,
            //     1.0 / self.frame_delta().as_secs_f32()
            // );
            return false;
        }

        // Did we hit a maximum of simulation steps and want to introduce lag instead?
        if let TimeConfiguration::RealtimeRenderingFixedSimulationStep { .. } = self.config {
            if self.num_simulation_steps_this_frame * simulation_delta > max_total_step_per_frame {
                // We heuristically don't drop all lost simulation frames. This avoids oscillating between realtime and offline
                // which is caused by our frame deltas being influenced by work from a couple of cpu frames ago (due gpu/cpu sync)
                self.accepted_simulation_to_render_lag += residual_time.mul_f32(0.75);
                // println!(
                //     "lagtime {}, fps {}",
                //     self.num_simulation_steps_this_frame,
                //     1.0 / self.frame_delta().as_secs_f32()
                // );
                return false;
            }
        }

        self.num_simulation_steps_this_frame += 1;
        self.num_simulation_steps += 1;
        true
    }

    fn simulation_delta(&self) -> Duration {
        match self.config {
            TimeConfiguration::RealtimeRenderingFixedSimulationStepCountPerFrame(count_per_frame) => self.current_frame_delta / count_per_frame,
            TimeConfiguration::RealtimeRenderingFixedSimulationStep { simulation_delta, .. } => simulation_delta,
        }
        .max(Duration::from_nanos(1))
    }

    pub fn frame_delta(&self) -> Duration {
        self.current_frame_delta
    }

    // Duration of the previous frame. (this is not necessarily equal to the frame time delta!)
    pub fn duration_for_last_frame(&self) -> Duration {
        self.time_since_last_frame_submitted
    }

    // Total render time. (equal to real time if not configured otherwise!)
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

    pub fn fill_uniform_buffer(&self) -> FrameTimeUniformBufferContent {
        FrameTimeUniformBufferContent {
            total_passed: self.total_rendered_time.as_secs_f32(),
            frame_delta: self.current_frame_delta.as_secs_f32(),
            total_simulated_time: self.total_simulated_time.as_secs_f32(),
            simulation_delta: self.simulation_delta().as_secs_f32(),
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
