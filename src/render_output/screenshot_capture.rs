use futures::*;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::mpsc::{channel, Receiver, Sender};

use crate::utils::round_to_multiple;

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

                let imgbuf = {
                    let screenshot_buffer_slice = buffer.slice(..);
                    let padded_buffer = screenshot_buffer_slice.get_mapped_range();
                    let padded_row_size = ScreenshotCapture::screenshot_buffer_bytes_per_padded_row(resolution);
                    let mut imgbuf = image::ImageBuffer::<image::Rgb<u8>, std::vec::Vec<_>>::new(resolution.width, resolution.height);
                    for (image_row, buffer_chunk) in imgbuf.rows_mut().zip(padded_buffer.chunks(padded_row_size)) {
                        for (image_pixel, buffer_pixel) in image_row.zip(buffer_chunk.chunks(4)) {
                            *image_pixel = image::Rgb([buffer_pixel[0], buffer_pixel[1], buffer_pixel[2]]);
                        }
                    }
                    imgbuf
                };

                buffer.unmap();
                completion_sender_clone.send(buffer).unwrap();
                imgbuf.save(target_path.clone()).unwrap();

                info!("Wrote screenshot to {:?} (took {:?})", target_path, start_time.elapsed());
            });
            return None;
        }
        return Some(self);
    }
}

pub struct ScreenshotCapture {
    unused_screenshot_buffers: Vec<wgpu::Buffer>,
    pending_screenshots: VecDeque<PendingScreenshot>,
    screenshot_completion_receiver: Receiver<wgpu::Buffer>,
    screenshot_completion_sender: Sender<wgpu::Buffer>,

    resolution: winit::dpi::PhysicalSize<u32>,
}

// This seems like an excessively high number, but it allows us to stream out video even if the picture format is heavier
const NUM_SCREENSHOT_BUFFERS: usize = 10;

impl ScreenshotCapture {
    pub fn new(device: &wgpu::Device, resolution: winit::dpi::PhysicalSize<u32>) -> Self {
        let mut unused_screenshot_buffers = Vec::new();
        for i in 0..NUM_SCREENSHOT_BUFFERS {
            unused_screenshot_buffers.push(device.create_buffer(&wgpu::BufferDescriptor {
                size: ScreenshotCapture::screenshot_buffer_bytes_per_padded_row(resolution) as u64 * resolution.height as u64,
                usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
                label: Some(&format!("Buffer: Screenshot readback buffer {}", i)),
                mapped_at_creation: false,
            }));
        }
        let (screenshot_completion_sender, screenshot_completion_receiver) = channel();

        ScreenshotCapture {
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

    pub fn capture_screenshot(&mut self, path: &Path, backbuffer: &wgpu::Texture, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        if self.unused_screenshot_buffers.len() == 0 {
            device.poll(wgpu::Maintain::Poll);
            self.process_pending_screenshots();

            if self.unused_screenshot_buffers.len() == 0 {
                warn!("No more unused screenshot buffers available. Waiting for GPU/writer to catch up and draining screenshot queue...");
                while self.unused_screenshot_buffers.len() == 0 {
                    std::thread::yield_now();
                    device.poll(wgpu::Maintain::Poll);
                    self.process_pending_screenshots();
                }
            }
        }
        let buffer = self.unused_screenshot_buffers.pop().unwrap();

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &backbuffer,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::ImageCopyBuffer {
                buffer: &buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: std::num::NonZeroU32::new(ScreenshotCapture::screenshot_buffer_bytes_per_padded_row(self.resolution) as u32),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: self.resolution.width,
                height: self.resolution.height,
                depth_or_array_layers: 1,
            },
        );

        self.pending_screenshots.push_back(PendingScreenshot {
            copy_operation: None,
            buffer,
            target_path: path.into(),
        });
    }
}
