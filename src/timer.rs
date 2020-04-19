use std::time::{Duration, Instant};

#[allow(dead_code)]
pub enum TimeConfiguration {
    // Every frame has a fixed number of steps. (simulation step length varies with frame time!)
    RealtimeRenderingFixedSimulationStepCountPerFrame(u32),
    // Given fixed timestep.
    RealtimeRenderingFixedSimulationStep {
        simulation_delta: Duration,
        // maximum number of steps per frame
        max_total_step_per_frame: Duration,
    },
    // Special recording mode that keeps simulation & rendering time length the same, but still keeps correct statistics.
    Recording {
        delta: Duration,
    },
}

// There is three dependent clocks:
// * real time
//      that's the watch on your wrist
// * render time
//      same as on your watch if you're not recording (!) but in chunks
// * simulation time
//      tries to keep up with render time but in different chunks and may start to drop steps
//
// Note that since our simulation is all on GPU it doesn't make sense to take timings around simulation steps on CPU!
pub struct Timer {
    config: TimeConfiguration,

    // real time measures
    timestamp_last_frame: Instant,
    last_frame_duration: Duration,

    // todo: Keep statistics over the last couple of render frames for display of smooth timings

    // render time measures
    render_total_passed: Duration,

    // simulation time
    num_simulation_steps_this_frame: u32,
    simulation_total_passed: Duration,
    simulation_current_frame_passed: Duration,
    accepted_simulation_to_render_lag: Duration, // time lost that we don't plan on catching up anymore
}

impl Timer {
    pub fn new(config: TimeConfiguration) -> Timer {
        Timer {
            config,

            timestamp_last_frame: Instant::now(),
            last_frame_duration: Duration::from_millis(0),

            render_total_passed: Duration::from_millis(0),

            num_simulation_steps_this_frame: 0,
            simulation_total_passed: Duration::from_millis(0),
            simulation_current_frame_passed: Duration::from_millis(0),
            accepted_simulation_to_render_lag: Duration::from_millis(0),
        }
    }

    pub fn on_frame_submitted(&mut self) {
        self.last_frame_duration = self.timestamp_last_frame.elapsed();
        self.render_total_passed += self.frame_delta();
        self.timestamp_last_frame = std::time::Instant::now();
        self.simulation_current_frame_passed = Duration::from_millis(0);
        self.num_simulation_steps_this_frame = 0;
    }

    pub fn simulation_step_loop(&mut self) -> bool {
        let simulation_delta = self.simulation_delta();
        if self.num_simulation_steps_this_frame > 0 {
            self.simulation_total_passed += simulation_delta;
            self.simulation_current_frame_passed += simulation_delta;
        }

        // simulation time shouldn't advance faster than render time
        let residual_time = self
            .render_total_passed
            .checked_sub(self.simulation_total_passed + self.accepted_simulation_to_render_lag)
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
        if let TimeConfiguration::RealtimeRenderingFixedSimulationStep {
            max_total_step_per_frame, ..
        } = self.config
        {
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
        true
    }

    fn simulation_delta(&self) -> Duration {
        match self.config {
            TimeConfiguration::RealtimeRenderingFixedSimulationStepCountPerFrame(count_per_frame) => self.frame_delta() / count_per_frame,
            TimeConfiguration::RealtimeRenderingFixedSimulationStep { simulation_delta, .. } => simulation_delta,
            TimeConfiguration::Recording { delta } => delta,
        }
        .max(Duration::from_nanos(1))
    }

    pub fn frame_delta(&self) -> Duration {
        match self.config {
            TimeConfiguration::RealtimeRenderingFixedSimulationStepCountPerFrame(_) => self.last_frame_duration,
            TimeConfiguration::RealtimeRenderingFixedSimulationStep { .. } => self.last_frame_duration,
            TimeConfiguration::Recording { delta } => delta,
        }
        .max(Duration::from_nanos(1))
    }

    // Time passed since startup. Kept constant for the duration of this frame.
    // pub fn time_since_start(&self) -> Duration {
    //     self.time_since_startup
    // }

    // Duration of the previous frame. (this is not necessarily equal to the time delta!)
    pub fn frame_duration(&self) -> Duration {
        self.last_frame_duration
    }

    pub fn fill_uniform_buffer(&self) -> FrameTimeUniformBufferContent {
        FrameTimeUniformBufferContent {
            total_passed: self.render_total_passed.as_secs_f32(),
            frame_delta: self.frame_delta().as_secs_f32(),
            simulation_total_passed: self.simulation_total_passed.as_secs_f32(),
            simulation_delta: self.simulation_delta().as_secs_f32().min(1.0 / 60.0),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct FrameTimeUniformBufferContent {
    pub total_passed: f32,            // How much time has passed on the rendering clock since rendering started.
    pub frame_delta: f32,             // How long a previous frame took in seconds.
    pub simulation_total_passed: f32, // How much time has passed in the simulation *excluding any steps in the current frame*
    pub simulation_delta: f32,        // How much we're advancing the simulation for each step in the current frame.
}
