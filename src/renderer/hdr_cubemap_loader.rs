use image::hdr::{HdrDecoder, Rgbe8Pixel};
use std::{fs::File, io::BufReader, path::Path};

// Loads cubemap in rgbe format
pub fn load_cubemap(path: &Path, device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::TextureView {
    let filenames = ["px.hdr", "nx.hdr", "py.hdr", "ny.hdr", "pz.hdr", "nz.hdr"];

    let mut cubemap = None;
    let mut resolution: u32 = 0;

    for (i, filename) in filenames.iter().enumerate() {
        info!("loading cubemap face {}..", i);

        let file_reader = BufReader::new(File::open(path.join(filename)).unwrap());
        let decoder = HdrDecoder::new(file_reader).unwrap();
        let metadata = decoder.metadata();

        if metadata.height != metadata.width {
            panic!("cubemap face width not equal height");
        }

        if let &None = &cubemap {
            resolution = metadata.width;
            cubemap = Some(device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Cubemap"),
                size: wgpu::Extent3d {
                    width: resolution,
                    height: resolution,
                    depth: 6,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
            }));
        }

        if resolution != metadata.width {
            panic!("all cubemap faces need to have the same resolution");
        }

        let image_data = decoder.read_image_native().unwrap();
        let image_data_raw =
            unsafe { std::slice::from_raw_parts(image_data.as_ptr() as *const u8, image_data.len() * std::mem::size_of::<Rgbe8Pixel>()) };

        queue.write_texture(
            wgpu::TextureCopyView {
                texture: &cubemap.as_ref().unwrap(),
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: i as u32 },
            },
            image_data_raw,
            wgpu::TextureDataLayout {
                offset: 0,
                bytes_per_row: std::mem::size_of::<Rgbe8Pixel>() as u32 * resolution,
                rows_per_image: 0,
            },
            wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth: 1,
            },
        );
    }

    cubemap.unwrap().create_view(&wgpu::TextureViewDescriptor {
        label: None,
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..wgpu::TextureViewDescriptor::default()
    })
}
