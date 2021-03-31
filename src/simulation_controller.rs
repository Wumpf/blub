use crate::scene::Scene;
use crate::{
    timer::{SimulationStepResult, Timer},
    wgpu_utils::pipelines::PipelineManager,
};
use std::time::{Duration, Instant};
use wgpu_profiler::GpuProfiler;

// The simulation controller orchestrates simulation steps.
// It holds the central timer and as such is responsible for glueing rendering frames and simulation together.
#[derive(PartialEq, Eq, Clone, Copy)]
pub enum SimulationControllerStatus {
    Realtime,
    RecordingWithFixedFrameLength(Duration),
    FastForward(Duration),
    Paused,
}

pub struct SimulationController {
    timer: Timer,
    computation_time_last_fast_forward: Duration,
    simulation_steps_per_second: u64,
    status: SimulationControllerStatus,
    pub simulation_stop_time: Duration,
    pub time_scale: f32,
}

// The maximum length of a single step we're willing to do in a single frame.
// If we need to compute more steps than this in a single frame, we give up and slow down the sim.
// -> this is correlated but not equal to the minimum target framerate.
const MAX_STEP_COMPUTATION_PER_FRAME: f64 = 1.0 / 50.0; // i.e. give up on keeping realtime if simulation alone would lead to 30fps

fn delta_from_steps_per_second(steps_per_second: u64) -> Duration {
    Duration::from_nanos(1000 * 1000 * 1000 / steps_per_second)
}

impl SimulationController {
    pub fn new() -> Self {
        const DEFAULT_SIMULATION_STEPS_PER_SECOND: u64 = 120;

        SimulationController {
            status: SimulationControllerStatus::Realtime,
            simulation_stop_time: Duration::from_secs(60 * 60), // (an hour)
            simulation_steps_per_second: DEFAULT_SIMULATION_STEPS_PER_SECOND,
            timer: Timer::new(delta_from_steps_per_second(DEFAULT_SIMULATION_STEPS_PER_SECOND)),
            computation_time_last_fast_forward: Default::default(),
            time_scale: 1.0,
        }
    }

    pub fn timer(&self) -> &Timer {
        &self.timer
    }

    pub fn on_frame_submitted(&mut self) {
        self.timer.on_frame_submitted(self.time_scale);
    }

    pub fn computation_time_last_fast_forward(&self) -> Duration {
        self.computation_time_last_fast_forward
    }

    pub fn simulation_steps_per_second(&self) -> u64 {
        self.simulation_steps_per_second
    }

    pub fn status(&self) -> SimulationControllerStatus {
        self.status
    }

    pub fn pause_or_resume(&mut self) {
        if self.status == SimulationControllerStatus::Paused {
            self.status = SimulationControllerStatus::Realtime;
        } else {
            self.status = SimulationControllerStatus::Paused;
        }
    }

    pub fn start_recording_with_fixed_frame_length(&mut self, frames_per_second: f64) {
        self.status = SimulationControllerStatus::RecordingWithFixedFrameLength(Duration::from_secs_f64(1.0 / frames_per_second));
    }

    pub fn set_simulation_steps_per_second(&mut self, simulation_steps_per_second: u64) {
        self.simulation_steps_per_second = simulation_steps_per_second;
        self.timer
            .set_simulation_delta(delta_from_steps_per_second(self.simulation_steps_per_second));
    }

    pub fn restart(&mut self) {
        self.timer = Timer::new(delta_from_steps_per_second(self.simulation_steps_per_second));
    }

    // A single fast forward operation is technically just a "very long frame".
    // However, since we need to give the GPU some breathing space it's handled in a different way (-> TDR).
    // Note that we assume that this never happens for realtime & recording, but it well could once a single simulation + render step takes longer than TDR time.
    pub fn fast_forward_steps(
        &mut self,
        simulation_jump_length: Duration,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &mut Scene,
        pipeline_manager: &PipelineManager,
        global_bind_group: &wgpu::BindGroup,
    ) {
        // After every batch we wait until the gpu is done.
        // This is not optimal for performance but is necessary because:
        // * avoid overloading gpu/driver command queue (we typically finish recording much quicker than gpu is doing the simulation)
        // * make it possible to readback simulation data
        // Doing wait per step introduces too much stalling, by batching we're going a middle ground.
        //
        // Ideally we would like to never wait until the queue is flushed (i.e. have n steps in flight), but this is hard to do with wgpu!
        const MAX_FAST_FORWARD_SIMULATION_BATCH_SIZE: usize = 16;

        self.status = SimulationControllerStatus::FastForward(simulation_jump_length);

        // re-use stopping standard stopping mechanism to halt the simulation
        let previous_simulation_end = self.simulation_stop_time;
        // jump at least one simulation step, makes for easier ui code
        self.simulation_stop_time = self.timer.total_simulated_time() + simulation_jump_length.max(self.timer.simulation_delta());
        let num_expected_steps = simulation_jump_length.max(self.timer.simulation_delta()).as_nanos() / self.timer.simulation_delta().as_nanos();

        let mut dummy_profiler = GpuProfiler::new(1, 0.0);
        dummy_profiler.enable_timer = false;
        dummy_profiler.enable_debug_marker = false;

        self.start_simulation_frame();
        {
            let start_time = Instant::now();
            let mut num_steps_finished = 0;
            while let SimulationControllerStatus::FastForward(..) = self.status {
                let mut batch_size = MAX_FAST_FORWARD_SIMULATION_BATCH_SIZE;
                {
                    for i in 0..MAX_FAST_FORWARD_SIMULATION_BATCH_SIZE {
                        if !self.single_step(scene, device, queue, pipeline_manager, &mut dummy_profiler, global_bind_group) {
                            batch_size = i;
                            break;
                        }
                    }
                }
                device.poll(wgpu::Maintain::Wait);
                num_steps_finished += batch_size;
                info!(
                    "simulation fast forwarding batch finished (progress {}/{})",
                    num_steps_finished, num_expected_steps
                );
            }
            self.computation_time_last_fast_forward = start_time.elapsed();
        }
        self.timer.on_frame_submitted(1.0);
        self.timer.force_frame_delta(Duration::from_secs(0));
        self.simulation_stop_time = previous_simulation_end;

        info!(
            "Fast forward of {:?} took {:?} to compute",
            simulation_jump_length, self.computation_time_last_fast_forward
        );
    }

    pub fn frame_steps(
        &mut self,
        scene: &mut Scene,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipeline_manager: &PipelineManager,
        profiler: &mut GpuProfiler,
        global_bind_group: &wgpu::BindGroup,
    ) {
        if !self.start_simulation_frame() {
            return;
        }

        while self.single_step(scene, device, queue, pipeline_manager, profiler, global_bind_group) {}
    }

    fn start_simulation_frame(&mut self) -> bool {
        match self.status {
            SimulationControllerStatus::Realtime => {}
            SimulationControllerStatus::RecordingWithFixedFrameLength(frame_length) => {
                self.timer.force_frame_delta(frame_length);
            }
            SimulationControllerStatus::FastForward(frame_length) => {
                self.timer.force_frame_delta(frame_length);
            }
            SimulationControllerStatus::Paused => {
                self.timer.skip_simulation_frame();
                return false;
            }
        };
        return true;
    }

    fn single_step<'a>(
        &mut self,
        scene: &'a mut Scene,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipeline_manager: &'a PipelineManager,
        profiler: &mut GpuProfiler,
        global_bind_group: &wgpu::BindGroup,
    ) -> bool {
        // frame drops are only relevant in realtime mode.
        let max_total_step_per_frame = if self.status == SimulationControllerStatus::Realtime {
            Duration::from_secs_f64(MAX_STEP_COMPUTATION_PER_FRAME)
        } else {
            Duration::from_secs(u64::MAX)
        };

        if self.timer.total_simulated_time() + self.timer.simulation_delta() > self.simulation_stop_time {
            self.status = SimulationControllerStatus::Paused;
            return false;
        }

        if self.timer.simulation_frame_loop(max_total_step_per_frame) == SimulationStepResult::PerformStepAndCallAgain {
            scene.step(&self.timer, device, profiler, pipeline_manager, queue, global_bind_group);
            return true;
        }
        return false;
    }
}
