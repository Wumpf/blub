pub struct Screen {
    pub resolution: winit::dpi::PhysicalSize<u32>,
    swap_chain: wgpu::SwapChain,
    depth_view: wgpu::TextureView,
}

impl Screen {
    pub const FORMAT_BACKBUFFER: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;
    pub const FORMAT_DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    pub fn new(device: &wgpu::Device, window_surface: &wgpu::Surface, resolution: winit::dpi::PhysicalSize<u32>) -> Self {
        info!("creating screen with {:?}", resolution);

        let swap_chain = device.create_swap_chain(
            window_surface,
            &wgpu::SwapChainDescriptor {
                usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
                format: Self::FORMAT_BACKBUFFER,
                width: resolution.width,
                height: resolution.height,
                present_mode: wgpu::PresentMode::Immediate,
            },
        );
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Texture: Screen DepthBuffer"),
            size: wgpu::Extent3d {
                width: resolution.width,
                height: resolution.height,
                depth: 1,
            },
            array_layer_count: 1,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::FORMAT_DEPTH,
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        });

        Screen {
            resolution,
            swap_chain,
            depth_view: depth_texture.create_default_view(),
        }
    }

    pub fn aspect_ratio(&self) -> f32 {
        self.resolution.width as f32 / self.resolution.height as f32
    }

    pub fn get_next_frame(&mut self) -> (wgpu::SwapChainOutput, &wgpu::TextureView) {
        (self.swap_chain.get_next_texture().unwrap(), &self.depth_view)
    }
}
