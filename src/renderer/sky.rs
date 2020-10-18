pub struct Sky {
    cubemap_view: wgpu::TextureView,
}

impl Sky {
    pub fn new(device: &wgpu::Device, resolution: u32) -> Self {
        let cubemap = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Sky Cubemap"),
            size: wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg11b10Float,
            usage: wgpu::TextureUsage::SAMPLED,
        });

        let cubemap_view = cubemap.create_view(&wgpu::TextureViewDescriptor {
            label: None,
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..wgpu::TextureViewDescriptor::default()
        });

        Sky { cubemap_view }
    }

    pub fn cubemap_view(&self) -> &wgpu::TextureView {
        &self.cubemap_view
    }
}
