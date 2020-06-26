use super::screen::Screen;
use std::path::{Path, PathBuf};

pub struct ScreenshotRecorder {
    next_regular_screenshot_index: usize,
    scheduled_screenshot: Option<PathBuf>,

    next_recording_screenshot_index: usize,
    recording_output_dir: Option<PathBuf>,
}

impl ScreenshotRecorder {
    pub fn new() -> Self {
        let mut next_regular_screenshot_index = 0;
        for i in 1..usize::MAX {
            if !Self::regular_screenshot_path(i).exists() {
                next_regular_screenshot_index = i;
                break;
            }
        }

        ScreenshotRecorder {
            next_regular_screenshot_index,
            scheduled_screenshot: None,

            next_recording_screenshot_index: 0,
            recording_output_dir: None,
        }
    }

    fn regular_screenshot_path(index: usize) -> PathBuf {
        PathBuf::from(format!("screenshot{}.png", index))
    }

    pub fn start_next_recording(&mut self) {
        for i in 0..usize::MAX {
            let recording_output_dir = PathBuf::from(format!("recording{}", i));
            if !recording_output_dir.exists() {
                self.start_recording(&recording_output_dir);
                break;
            }
        }
    }

    fn start_recording(&mut self, recording_output_dir: &Path) {
        std::fs::create_dir(&recording_output_dir).unwrap();
        self.next_recording_screenshot_index = 0;
        self.recording_output_dir = Some(recording_output_dir.into());
    }

    pub fn stop_recording(&mut self) {
        self.recording_output_dir = None;
    }

    pub fn schedule_next_screenshot(&mut self) {
        self.schedule_screenshot(&Self::regular_screenshot_path(self.next_regular_screenshot_index));
        self.next_regular_screenshot_index += 1;
    }

    fn schedule_screenshot(&mut self, path: &Path) {
        self.scheduled_screenshot = Some(path.into());
    }

    pub fn capture_screenshot(&mut self, screen: &mut Screen, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        if let Some(ref scheduled_screenshot) = self.scheduled_screenshot {
            screen.capture_screenshot(&scheduled_screenshot, device, encoder);
        }
        if let Some(ref recording_output_dir) = self.recording_output_dir {
            screen.capture_screenshot(
                &recording_output_dir.join(format!("screenshot{}.png", self.next_recording_screenshot_index)),
                device,
                encoder,
            );
            self.next_recording_screenshot_index += 1;
        }

        self.scheduled_screenshot = None;
    }
}
