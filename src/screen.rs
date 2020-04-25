use crate::wgpu_utils::binding_builder::*;
use crate::wgpu_utils::shader::*;
use crate::wgpu_utils::*;
use std::path::{Path, PathBuf};

pub struct Screen {
    pub resolution: winit::dpi::PhysicalSize<u32>,
    swap_chain: wgpu::SwapChain,

    backbuffer: wgpu::Texture,
    backbuffer_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,

    read_backbuffer_bind_group: wgpu::BindGroup,
    copy_to_swapchain_pipeline: wgpu::RenderPipeline,

    screenshot_buffer: wgpu::Buffer,
    next_screenshot_file: PathBuf,
}

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
                present_mode: wgpu::PresentMode::Immediate,
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
            array_layer_count: 1,
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
            array_layer_count: 1,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::FORMAT_DEPTH,
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        });

        let screenshot_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            size: (resolution.width * resolution.height) as u64 * std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
            label: Some("Buffer: Screenshot readback"),
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
            rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: wgpu::CullMode::None,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
            }),
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            color_states: &[wgpu::ColorStateDescriptor {
                format: super::Screen::FORMAT_SWAPCHAIN,
                color_blend: wgpu::BlendDescriptor::REPLACE,
                alpha_blend: wgpu::BlendDescriptor::REPLACE,
                write_mask: wgpu::ColorWrite::ALL,
            }],
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

            screenshot_buffer,
            next_screenshot_file: PathBuf::default(),
        }
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

    pub fn start_frame(&mut self) -> wgpu::SwapChainOutput {
        self.swap_chain.get_next_texture().unwrap()
    }

    pub fn take_screenshot(&mut self, encoder: &mut wgpu::CommandEncoder, path: &Path) {
        encoder.copy_texture_to_buffer(
            wgpu::TextureCopyView {
                texture: &self.backbuffer,
                mip_level: 0,
                array_layer: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::BufferCopyView {
                buffer: &self.screenshot_buffer,
                offset: 0,
                bytes_per_row: std::mem::size_of::<u32>() as u32 * self.resolution.width,
                rows_per_image: 0,
            },
            wgpu::Extent3d {
                width: self.resolution.width,
                height: self.resolution.height,
                depth: 1,
            },
        );
        self.next_screenshot_file = path.into();
    }

    pub fn copy_to_swapchain(&mut self, output: &wgpu::SwapChainOutput, encoder: &mut wgpu::CommandEncoder) {
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

        render_pass.set_pipeline(&self.copy_to_swapchain_pipeline);
        render_pass.set_bind_group(0, &self.read_backbuffer_bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }

    pub fn end_frame(&mut self, device: &wgpu::Device, _output: wgpu::SwapChainOutput) {
        if self.next_screenshot_file == PathBuf::default() {
            return;
        }

        let buffer_future = self.screenshot_buffer.map_read(
            0,
            (self.resolution.width * self.resolution.height) as u64 * std::mem::size_of::<u32>() as u64,
        );

        // TODO: This is the worst possible way to deal with this. Should do Polls on frame starts and keep several buffers in order to never block on this.
        // Since we want to do video rendering here, we really should avoid any full stop on gpu rendering!
        device.poll(wgpu::Maintain::Wait);

        // Write the buffer as a PNG
        let start_time = std::time::Instant::now();
        let mapping = futures::executor::block_on(buffer_future).unwrap();
        let mut png_encoder = png::Encoder::new(
            std::fs::File::create(&self.next_screenshot_file).unwrap(),
            self.resolution.width,
            self.resolution.height,
        );
        png_encoder.set_depth(png::BitDepth::Eight);
        png_encoder.set_color(png::ColorType::RGBA);
        png_encoder.write_header().unwrap().write_image_data(mapping.as_slice()).unwrap();

        info!("Wrote screenshot to {:?} (took {:?})", self.next_screenshot_file, start_time.elapsed());
        self.next_screenshot_file = PathBuf::default();
    }
}
