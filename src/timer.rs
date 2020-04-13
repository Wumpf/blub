use std::time::{Duration, Instant};

pub enum SimulationTimeConfiguration {
    OneStepPerFrame,
}

pub struct Timer {
    timestamp_startup: Instant,
    timestamp_last_frame: Instant,
    time_since_startup: Duration,
    last_frame_duration: Duration,
    // todo: Keep statistics over the last couple of frames for display of smooth framerate
    simulation_config: SimulationTimeConfiguration,
    simulation_total_passed: Duration,
}

impl Timer {
    pub fn new(simulation_config: SimulationTimeConfiguration) -> Timer {
        Timer {
            timestamp_startup: Instant::now(),
            timestamp_last_frame: Instant::now(),
            time_since_startup: Duration::from_millis(0),
            last_frame_duration: Duration::from_millis(0),

            simulation_config,
            simulation_total_passed: Duration::from_millis(0),
        }
    }

    pub fn on_frame_submitted(&mut self) {
        self.time_since_startup = self.timestamp_startup.elapsed();
        self.last_frame_duration = self.timestamp_last_frame.elapsed();
        self.timestamp_last_frame = std::time::Instant::now();
    }

    pub fn on_simulation_step_completed(&mut self) {
        self.simulation_total_passed += self.simulation_delta();
    }

    fn simulation_delta(&self) -> Duration {
        match self.simulation_config {
            SimulationTimeConfiguration::OneStepPerFrame => self.last_frame_duration,
        }
    }

    // Time passed since startup. Kept constant for the duration of this frame.
    // pub fn time_since_start(&self) -> Duration {
    //     self.time_since_startup
    // }

    // Duration of the previous frame.
    pub fn frame_delta_time(&self) -> Duration {
        self.last_frame_duration
    }

    pub fn fill_uniform_buffer(&self) -> FrameTimeUniformBufferContent {
        FrameTimeUniformBufferContent {
            total_passed: self.time_since_startup.as_secs_f32(),
            frame_delta: self.last_frame_duration.as_secs_f32().max(std::f32::EPSILON),
            simulation_total_passed: self.simulation_total_passed.as_secs_f32(),
            simulation_delta: self.simulation_delta().as_secs_f32().max(std::f32::EPSILON),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct FrameTimeUniformBufferContent {
    pub total_passed: f32,            // How much time has passed in the real world since rendering started.
    pub frame_delta: f32,             // How long a previous frame took in seconds
    pub simulation_total_passed: f32, // How much time has passed in the simulation *excluding any steps in the current frame*
    pub simulation_delta: f32,        // How much we're advancing the simulation for each step in the current frame.
}
