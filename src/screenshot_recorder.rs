use futures::*;
use std::collections::VecDeque;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::mpsc::{channel, Receiver, Sender};

struct PendingScreenshot {
    copy_operation: Option<Pin<Box<dyn Future<Output = std::result::Result<(), wgpu::BufferAsyncError>>>>>,
    buffer: wgpu::Buffer,
    target_path: PathBuf,
}

impl PendingScreenshot {
    pub fn spawn_write_thread_if_ready(
        mut self,
        resolution: winit::dpi::PhysicalSize<u32>,
        completion_sender: &Sender<wgpu::Buffer>,
    ) -> Option<PendingScreenshot> {
        if self.copy_operation.is_none() {
            let screenshot_buffer_slice = self.buffer.slice(..);
            self.copy_operation = Some(screenshot_buffer_slice.map_async(wgpu::MapMode::Read).boxed());
        }

        let val = (&mut self.copy_operation.as_mut().unwrap()).now_or_never();
        if val.is_some() {
            let buffer = self.buffer;
            let target_path = self.target_path;
            let completion_sender_clone = completion_sender.clone();

            std::thread::spawn(move || {
                let start_time = std::time::Instant::now();
                let mut png_encoder = png::Encoder::new(std::fs::File::create(&target_path).unwrap(), resolution.width, resolution.height);
                png_encoder.set_depth(png::BitDepth::Eight);
                png_encoder.set_color(png::ColorType::RGBA);
                let mut png_writer = png_encoder
                    .write_header()
                    .unwrap()
                    .into_stream_writer_with_size(ScreenshotRecorder::screenshot_buffer_bytes_per_row(resolution));

                let screenshot_buffer_slice = buffer.slice(..);
                let padded_buffer = screenshot_buffer_slice.get_mapped_range().to_vec();
                for chunk in padded_buffer.chunks(ScreenshotRecorder::screenshot_buffer_bytes_per_padded_row(resolution) as usize) {
                    png_writer
                        .write(&chunk[..ScreenshotRecorder::screenshot_buffer_bytes_per_row(resolution)])
                        .unwrap();
                }
                buffer.unmap();
                completion_sender_clone.send(buffer).unwrap();
                png_writer.finish().unwrap();

                info!("Wrote screenshot to {:?} (took {:?})", target_path, start_time.elapsed());
            });
            return None;
        }
        return Some(self);
    }
}

pub struct ScreenshotRecorder {
    unused_screenshot_buffers: Vec<wgpu::Buffer>,
    pending_screenshots: VecDeque<PendingScreenshot>,
    screenshot_completion_receiver: Receiver<wgpu::Buffer>,
    screenshot_completion_sender: Sender<wgpu::Buffer>,

    resolution: winit::dpi::PhysicalSize<u32>,
}

// This seems like an excessively high number, but it allows us to stream out video even if the picture format is heavier
const NUM_SCREENSHOT_BUFFERS: usize = 10;

impl ScreenshotRecorder {
    pub fn new(device: &wgpu::Device, resolution: winit::dpi::PhysicalSize<u32>) -> Self {
        let mut unused_screenshot_buffers = Vec::new();
        for i in 0..NUM_SCREENSHOT_BUFFERS {
            unused_screenshot_buffers.push(device.create_buffer(&wgpu::BufferDescriptor {
                size: ScreenshotRecorder::screenshot_buffer_bytes_per_padded_row(resolution) as u64 * resolution.height as u64,
                usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
                label: Some(&format!("Buffer: Screenshot readback buffer {}", i)),
                mapped_at_creation: false,
            }));
        }
        let (screenshot_completion_sender, screenshot_completion_receiver) = channel();

        ScreenshotRecorder {
            unused_screenshot_buffers,
            pending_screenshots: VecDeque::new(),
            screenshot_completion_receiver,
            screenshot_completion_sender,

            resolution,
        }
    }

    fn screenshot_buffer_bytes_per_row(resolution: winit::dpi::PhysicalSize<u32>) -> usize {
        resolution.width as usize * std::mem::size_of::<u32>()
    }

    fn screenshot_buffer_bytes_per_padded_row(resolution: winit::dpi::PhysicalSize<u32>) -> usize {
        round_to_multiple(
            Self::screenshot_buffer_bytes_per_row(resolution),
            wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize,
        )
    }

    pub fn process_pending_screenshots(&mut self) {
        if let Some(pending_screenshot) = self.pending_screenshots.pop_front() {
            if let Some(still_pending_screenshot) =
                pending_screenshot.spawn_write_thread_if_ready(self.resolution, &self.screenshot_completion_sender)
            {
                self.pending_screenshots.push_front(still_pending_screenshot);
            }
        }
        if let Ok(received_unused_buffer) = self.screenshot_completion_receiver.try_recv() {
            self.unused_screenshot_buffers.push(received_unused_buffer);
        }
    }

    pub fn wait_for_pending_screenshots(&mut self, device: &wgpu::Device) {
        while self.unused_screenshot_buffers.len() < NUM_SCREENSHOT_BUFFERS {
            device.poll(wgpu::Maintain::Poll);
            self.process_pending_screenshots();
            std::thread::yield_now();
        }
    }

    pub fn take_screenshot(&mut self, path: &Path, backbuffer: &wgpu::Texture, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        if self.unused_screenshot_buffers.len() == 0 {
            warn!("No more unused screenshot buffers available. Waiting for GPU/screenshot writer to catch up and draining screenshot queue...");
            while self.unused_screenshot_buffers.len() == 0 {
                device.poll(wgpu::Maintain::Poll);
                self.process_pending_screenshots();
                std::thread::yield_now();
            }
        }
        let buffer = self.unused_screenshot_buffers.pop().unwrap();

        encoder.copy_texture_to_buffer(
            wgpu::TextureCopyView {
                texture: &backbuffer,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::BufferCopyView {
                buffer: &buffer,
                layout: wgpu::TextureDataLayout {
                    offset: 0,
                    bytes_per_row: ScreenshotRecorder::screenshot_buffer_bytes_per_padded_row(self.resolution) as u32,
                    rows_per_image: 0,
                },
            },
            wgpu::Extent3d {
                width: self.resolution.width,
                height: self.resolution.height,
                depth: 1,
            },
        );

        self.pending_screenshots.push_back(PendingScreenshot {
            copy_operation: None,
            buffer,
            target_path: path.into(),
        });
    }
}

fn round_to_multiple(value: usize, multiple: usize) -> usize {
    (value + multiple - 1) / multiple * multiple
}
