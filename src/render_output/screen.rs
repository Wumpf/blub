use super::screenshot_capture::ScreenshotCapture;
use crate::wgpu_utils::binding_builder::*;
use crate::wgpu_utils::shader::*;
use crate::wgpu_utils::*;
use pipelines::*;
use std::{path::Path, rc::Rc};

pub struct Screen {
    resolution: winit::dpi::PhysicalSize<u32>,
    swap_chain: wgpu::SwapChain,
    present_mode: wgpu::PresentMode,

    backbuffer: wgpu::Texture,
    backbuffer_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,

    read_backbuffer_bind_group: wgpu::BindGroup,
    copy_to_swapchain_pipeline: RenderPipelineHandle,

    screenshot_capture: ScreenshotCapture,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ScreenUniformBufferContent {
    resolution: cgmath::Point2<f32>,
    resolution_inv: cgmath::Point2<f32>,
}

impl Screen {
    pub const FORMAT_BACKBUFFER: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
    const FORMAT_SWAPCHAIN: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;
    pub const FORMAT_DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
    pub const DEFAULT_PRESENT_MODE: wgpu::PresentMode = wgpu::PresentMode::Fifo;

    pub fn new(
        device: &wgpu::Device,
        window_surface: &wgpu::Surface,
        present_mode: wgpu::PresentMode,
        resolution: winit::dpi::PhysicalSize<u32>,
        shader_dir: &ShaderDirectory,
        pipeline_manager: &mut PipelineManager,
    ) -> Self {
        info!("creating screen with {:?}", resolution);

        let swap_chain = device.create_swap_chain(
            window_surface,
            &wgpu::SwapChainDescriptor {
                usage: wgpu::TextureUsage::RENDER_ATTACHMENT,
                format: Screen::FORMAT_SWAPCHAIN,
                width: resolution.width,
                height: resolution.height,
                present_mode,
            },
        );

        let size = wgpu::Extent3d {
            width: resolution.width,
            height: resolution.height,
            depth_or_array_layers: 1,
        };

        let backbuffer = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Texture: Backbuffer"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::FORMAT_BACKBUFFER,
            usage: wgpu::TextureUsage::RENDER_ATTACHMENT | wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_SRC,
        });
        let backbuffer_view = backbuffer.create_view(&Default::default());

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Texture: Screen DepthBuffer"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::FORMAT_DEPTH,
            usage: wgpu::TextureUsage::RENDER_ATTACHMENT,
        });

        let bind_group_layout = BindGroupLayoutBuilder::new()
            .next_binding_fragment(binding_glsl::texture2D())
            .create(device, "BindGroupLayout: Screen, Read Texture");
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Screen Swapchain Copy Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout.layout],
            push_constant_ranges: &[],
        });

        let read_backbuffer_bind_group = BindGroupBuilder::new(&bind_group_layout)
            .texture(&backbuffer_view)
            .create(device, "BindGroup: Read Backbuffer");

        let copy_to_swapchain_pipeline = pipeline_manager.create_render_pipeline(
            device,
            shader_dir,
            RenderPipelineCreationDesc::new(
                "Screen: Copy texture",
                Rc::new(pipeline_layout),
                Path::new("screentri.vert"),
                Path::new("copy_texture.frag"),
                Self::FORMAT_SWAPCHAIN,
                None,
            ),
        );

        Screen {
            resolution,
            swap_chain,
            present_mode,
            backbuffer,
            backbuffer_view,
            depth_view: depth_texture.create_view(&Default::default()),

            read_backbuffer_bind_group,
            copy_to_swapchain_pipeline,
            screenshot_capture: ScreenshotCapture::new(device, resolution),
        }
    }

    pub fn aspect_ratio(&self) -> f32 {
        self.resolution.width as f32 / self.resolution.height as f32
    }

    pub fn resolution(&self) -> winit::dpi::PhysicalSize<u32> {
        self.resolution
    }

    pub fn backbuffer(&self) -> &wgpu::TextureView {
        &self.backbuffer_view
    }

    pub fn depthbuffer(&self) -> &wgpu::TextureView {
        &self.depth_view
    }

    pub fn present_mode(&self) -> wgpu::PresentMode {
        self.present_mode
    }

    pub fn capture_screenshot(&mut self, path: &Path, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        self.screenshot_capture.capture_screenshot(path, &self.backbuffer, device, encoder);
    }

    pub fn start_frame(&mut self, device: &wgpu::Device, window_surface: &wgpu::Surface) -> wgpu::SwapChainTexture {
        // We assume here that any resizing has already been handled.
        // In that case it can still sometimes happen that the swap chain doesn't give a valid frame, e.g. after getting back from minimized state.
        // The problem usually goes away after recreating the swap chain.
        match self.swap_chain.get_current_frame() {
            Ok(frame) => frame.output,
            Err(_) => {
                info!(
                    "Failed to query current frame from swap chain. Recreating swap chain (resolution {:?}, present mode {:?})",
                    self.resolution, self.present_mode
                );
                self.swap_chain = device.create_swap_chain(
                    window_surface,
                    &wgpu::SwapChainDescriptor {
                        usage: wgpu::TextureUsage::RENDER_ATTACHMENT,
                        format: Screen::FORMAT_SWAPCHAIN,
                        width: self.resolution.width,
                        height: self.resolution.height,
                        present_mode: self.present_mode,
                    },
                );
                self.swap_chain.get_current_frame().unwrap().output
            }
        }
    }

    pub fn copy_to_swapchain(&mut self, output: &wgpu::SwapChainTexture, encoder: &mut wgpu::CommandEncoder, pipeline_manager: &PipelineManager) {
        // why this extra copy?
        // Webgpu doesn't allow us to do anything with the swapchain target but read from it!
        // That means that we can never take a screenshot.
        //
        // The only thing that could be done better here is to avoid this copy for frames that don't take screenshots.
        // However, this would require that backbuffer() gives out a different texture depending on whether this is a frame with or without screenshot.
        // Right now this is not possible since they have different formats. Could fix that, but all it safes us is this copy here (can't remove the buffer either)
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("copy to swapchain"),
            color_attachments: &[wgpu::RenderPassColorAttachment {
                view: &output.view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: true,
                },
            }],
            depth_stencil_attachment: None,
        });
        render_pass.set_pipeline(pipeline_manager.get_render(&self.copy_to_swapchain_pipeline));
        render_pass.set_bind_group(0, &self.read_backbuffer_bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }

    pub fn end_frame(&mut self, frame: wgpu::SwapChainTexture) {
        std::mem::drop(frame);
        self.screenshot_capture.process_pending_screenshots();
    }

    pub fn wait_for_pending_screenshots(&mut self, device: &wgpu::Device) {
        self.screenshot_capture.wait_for_pending_screenshots(device);
    }

    pub fn fill_global_uniform_buffer(&self) -> ScreenUniformBufferContent {
        ScreenUniformBufferContent {
            resolution: cgmath::point2(self.resolution.width as f32, self.resolution.height as f32),
            resolution_inv: cgmath::point2(1.0 / self.resolution.width as f32, 1.0 / self.resolution.height as f32),
        }
    }
}
