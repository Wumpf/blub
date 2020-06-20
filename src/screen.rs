use crate::wgpu_utils::binding_builder::*;
use crate::wgpu_utils::shader::*;
use crate::wgpu_utils::*;
use futures::*;
use pipelines::*;
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
    fn spawn_write_thread_if_ready(
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
                    .into_stream_writer_with_size(Screen::screenshot_buffer_bytes_per_row(resolution));

                let screenshot_buffer_slice = buffer.slice(..);
                let padded_buffer = screenshot_buffer_slice.get_mapped_range().to_vec();
                for chunk in padded_buffer.chunks(Screen::screenshot_buffer_bytes_per_padded_row(resolution) as usize) {
                    png_writer.write(&chunk[..Screen::screenshot_buffer_bytes_per_row(resolution)]).unwrap();
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

pub struct Screen {
    pub resolution: winit::dpi::PhysicalSize<u32>,
    swap_chain: wgpu::SwapChain,

    backbuffer: wgpu::Texture,
    backbuffer_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,

    read_backbuffer_bind_group: wgpu::BindGroup,
    copy_to_swapchain_pipeline: wgpu::RenderPipeline,

    unused_screenshot_buffers: Vec<wgpu::Buffer>,
    pending_screenshots: VecDeque<PendingScreenshot>,
    screenshot_completion_receiver: Receiver<wgpu::Buffer>,
    screenshot_completion_sender: Sender<wgpu::Buffer>,
}

// This seems like an excessively high number, but it allows us to stream out video even if the picture format is heavier
const NUM_SCREENSHOT_BUFFERS: usize = 10;

impl Screen {
    pub const FORMAT_BACKBUFFER: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
    pub const FORMAT_SWAPCHAIN: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;
    pub const FORMAT_DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    pub fn new(
        device: &wgpu::Device,
        window_surface: &wgpu::Surface,
        resolution: winit::dpi::PhysicalSize<u32>,
        shader_dir: &ShaderDirectory,
    ) -> Self {
        info!("creating screen with {:?}", resolution);

        let swap_chain = device.create_swap_chain(
            window_surface,
            &wgpu::SwapChainDescriptor {
                usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
                format: Screen::FORMAT_SWAPCHAIN,
                width: resolution.width,
                height: resolution.height,
                present_mode: wgpu::PresentMode::Mailbox,
            },
        );

        let size = wgpu::Extent3d {
            width: resolution.width,
            height: resolution.height,
            depth: 1,
        };

        let backbuffer = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Texture: Backbuffer"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::FORMAT_BACKBUFFER,
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT | wgpu::TextureUsage::COPY_SRC | wgpu::TextureUsage::SAMPLED,
        });
        let backbuffer_view = backbuffer.create_default_view();

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Texture: Screen DepthBuffer"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::FORMAT_DEPTH,
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        });

        let mut unused_screenshot_buffers = Vec::new();
        for i in 0..NUM_SCREENSHOT_BUFFERS {
            unused_screenshot_buffers.push(device.create_buffer(&wgpu::BufferDescriptor {
                size: Self::screenshot_buffer_bytes_per_padded_row(resolution) as u64 * resolution.height as u64,
                usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
                label: Some(&format!("Buffer: Screenshot readback buffer {}", i)),
                mapped_at_creation: false,
            }));
        }
        let (screenshot_completion_sender, screenshot_completion_receiver) = channel();

        let bind_group_layout = BindGroupLayoutBuilder::new()
            .next_binding_fragment(binding_glsl::texture2D())
            .create(device, "BindGroupLayout: Screen, Read Texture");
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&bind_group_layout.layout],
        });

        let read_backbuffer_bind_group = BindGroupBuilder::new(&bind_group_layout)
            .texture(&backbuffer_view)
            .create(device, "BindGroup: Read Backbuffer");

        let vs_module = shader_dir.load_shader_module(device, Path::new("screentri.vert")).unwrap();
        let fs_module = shader_dir.load_shader_module(device, Path::new("copy_texture.frag")).unwrap();
        let copy_to_swapchain_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            layout: &pipeline_layout,
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &vs_module,
                entry_point: SHADER_ENTRY_POINT_NAME,
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &fs_module,
                entry_point: SHADER_ENTRY_POINT_NAME,
            }),
            rasterization_state: Some(rasterization_state::culling_none()),
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            color_states: &[color_state::write_all(Self::FORMAT_SWAPCHAIN)],
            depth_stencil_state: None,
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: wgpu::IndexFormat::Uint16,
                vertex_buffers: &[],
            },
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        Screen {
            resolution,
            swap_chain,
            backbuffer,
            backbuffer_view,
            depth_view: depth_texture.create_default_view(),

            read_backbuffer_bind_group,
            copy_to_swapchain_pipeline,

            unused_screenshot_buffers,
            pending_screenshots: VecDeque::new(),
            screenshot_completion_receiver,
            screenshot_completion_sender,
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

    pub fn aspect_ratio(&self) -> f32 {
        self.resolution.width as f32 / self.resolution.height as f32
    }

    pub fn backbuffer(&self) -> &wgpu::TextureView {
        &self.backbuffer_view
    }

    pub fn depthbuffer(&self) -> &wgpu::TextureView {
        &self.depth_view
    }

    pub fn start_frame(&mut self) -> wgpu::SwapChainTexture {
        self.swap_chain.get_next_frame().unwrap().output
    }

    pub fn take_screenshot(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, path: &Path) {
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
                texture: &self.backbuffer,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::BufferCopyView {
                buffer: &buffer,
                layout: wgpu::TextureDataLayout {
                    offset: 0,
                    bytes_per_row: Self::screenshot_buffer_bytes_per_padded_row(self.resolution) as u32,
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

    pub fn copy_to_swapchain(&mut self, output: &wgpu::SwapChainTexture, encoder: &mut wgpu::CommandEncoder) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                attachment: &output.view,
                resolve_target: None,
                load_op: wgpu::LoadOp::Clear,
                store_op: wgpu::StoreOp::Store,
                clear_color: wgpu::Color::TRANSPARENT,
            }],
            depth_stencil_attachment: None,
        });
        render_pass.push_debug_group("screen - copy to swapchain");
        render_pass.set_pipeline(&self.copy_to_swapchain_pipeline);
        render_pass.set_bind_group(0, &self.read_backbuffer_bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }

    fn process_pending_screenshots(&mut self) {
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

    pub fn end_frame(&mut self, frame: wgpu::SwapChainTexture) {
        std::mem::drop(frame);
        self.process_pending_screenshots();
    }

    pub fn wait_for_pending_screenshots(&mut self, device: &wgpu::Device) {
        while self.unused_screenshot_buffers.len() < NUM_SCREENSHOT_BUFFERS {
            device.poll(wgpu::Maintain::Poll);
            self.process_pending_screenshots();
            std::thread::yield_now();
        }
    }
}

fn round_to_multiple(value: usize, multiple: usize) -> usize {
    (value + multiple - 1) / multiple * multiple
}
