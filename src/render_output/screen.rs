use super::screenshot_capture::ScreenshotCapture;
use crate::wgpu_utils::binding_builder::*;
use crate::wgpu_utils::shader::*;
use crate::wgpu_utils::*;
use pipelines::*;
use std::path::Path;

pub struct Screen {
    resolution: winit::dpi::PhysicalSize<u32>,
    swap_chain: wgpu::SwapChain,

    backbuffer: wgpu::Texture,
    backbuffer_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,

    read_backbuffer_bind_group: wgpu::BindGroup,
    copy_to_swapchain_pipeline: wgpu::RenderPipeline,

    screenshot_capture: ScreenshotCapture,
}

impl Screen {
    pub const FORMAT_BACKBUFFER: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
    const FORMAT_SWAPCHAIN: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;
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
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT | wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_SRC,
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

    pub fn capture_screenshot(&mut self, path: &Path, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        self.screenshot_capture.capture_screenshot(path, &self.backbuffer, device, encoder);
    }

    pub fn start_frame(&mut self) -> wgpu::SwapChainTexture {
        self.swap_chain.get_next_frame().unwrap().output
    }

    pub fn copy_to_swapchain(&mut self, output: &wgpu::SwapChainTexture, encoder: &mut wgpu::CommandEncoder) {
        // why this extra copy?
        // Webgpu doesn't allow us to do anything with the swapchain target but read from it!
        // That means that we can never take a screenshot.
        //
        // The only thing that could be done better here is to avoid this copy for frames that don't take screenshots.
        // However, this would require that backbuffer() gives out a different texture depending on whether this is a frame with or without screenshot.
        // Right now this is not possible since they have different formats. Could fix that, but all it safes us is this copy here (can't remove the buffer either)
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

    pub fn end_frame(&mut self, frame: wgpu::SwapChainTexture) {
        std::mem::drop(frame);
        self.screenshot_capture.process_pending_screenshots();
    }

    pub fn wait_for_pending_screenshots(&mut self, device: &wgpu::Device) {
        self.screenshot_capture.wait_for_pending_screenshots(device);
    }
}
