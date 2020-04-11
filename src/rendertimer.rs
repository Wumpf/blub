use std::time::{Duration, Instant};

pub struct RenderTimer {
    timestamp_startup: Instant,
    timestamp_last_frame: Instant,
    time_since_startup: Duration,
    last_frame_duration: Duration,
    // todo: Keep statistics over the last couple of frames for display of smooth framerate
}

impl RenderTimer {
    pub fn new() -> RenderTimer {
        RenderTimer {
            timestamp_startup: Instant::now(),
            timestamp_last_frame: Instant::now(),
            time_since_startup: Duration::from_millis(0),
            last_frame_duration: Duration::from_millis(16), // Zero sized frames could cause issues.
        }
    }

    pub fn on_frame_submitted(&mut self) {
        self.time_since_startup = self.timestamp_startup.elapsed();
        self.last_frame_duration = self.timestamp_last_frame.elapsed();
        self.timestamp_last_frame = std::time::Instant::now();
    }

    // Time passed since startup. Kept constant for the duration of this frame.
    pub fn time_since_start(&self) -> Duration {
        self.time_since_startup
    }

    // Duration of the previous frame.
    pub fn frame_delta_time(&self) -> Duration {
        self.last_frame_duration
    }
}
