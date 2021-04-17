use crate::{
    render_output::hdr_backbuffer::HdrBackbuffer,
    render_output::screen::Screen,
    wgpu_utils::uniformbuffer::PaddedVector3,
    wgpu_utils::{binding_builder::*, binding_glsl, pipelines::*, shader::ShaderDirectory, uniformbuffer::UniformBuffer},
};
use serde::Deserialize;
use std::{fs::File, io, io::BufReader, path::Path, rc::Rc};

// Data describing a scene.
#[derive(Deserialize)]
pub struct BackgroundConfig {
    pub dir_light_direction: cgmath::Vector3<f32>,
    pub dir_light_radiance: cgmath::Vector3<f32>,
    pub indirect_lighting_sh: [(f32, f32, f32); 9],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct LightingAndBackgroundUniformBufferContent {
    pub dir_light_direction: PaddedVector3,
    pub dir_light_radiance: PaddedVector3,
    pub indirect_lighting_sh: [((f32, f32, f32), f32); 9],
}
unsafe impl bytemuck::Pod for LightingAndBackgroundUniformBufferContent {}
unsafe impl bytemuck::Zeroable for LightingAndBackgroundUniformBufferContent {}

type LightingAndBackgroundUniformBuffer = UniformBuffer<LightingAndBackgroundUniformBufferContent>;

pub struct Background {
    pipeline: RenderPipelineHandle,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

mod cubemap_loader {
    use image::hdr::Rgbe8Pixel;
    use std::{
        fs::File,
        io::{Read, Write},
        num::NonZeroU32,
        path::{Path, PathBuf},
    };

    const CUBEMAP_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
    const CUBEMAP_FORMAT_BYTES_PER_PIXEL: u32 = std::mem::size_of::<Rgbe8Pixel>() as u32;

    fn get_cache_filename(path: &Path) -> PathBuf {
        path.join(format!(".raw_rgbe8_cubemap.cache"))
    }

    fn from_cache(path: &Path, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<wgpu::Texture, std::io::Error> {
        let cache_filename = get_cache_filename(path);
        info!("loading cubemap from cached raw file at {:?}", cache_filename);

        let mut image_data = Vec::new();
        let num_bytes_read = File::open(cache_filename)?.read_to_end(&mut image_data).unwrap();

        let resolution = f32::sqrt((num_bytes_read / 4 / 6) as f32) as u32;

        let cubemap = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Cubemap"),
            size: wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: CUBEMAP_FORMAT,
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
        });

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &cubemap,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            &image_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: NonZeroU32::new(CUBEMAP_FORMAT_BYTES_PER_PIXEL * resolution),
                rows_per_image: NonZeroU32::new(resolution),
            },
            wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: 6,
            },
        );

        Ok(cubemap)
    }

    // Loads cubemap in rgbe format
    fn from_hdr_faces(path: &Path, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<wgpu::Texture, std::io::Error> {
        let filenames = ["px.hdr", "nx.hdr", "py.hdr", "ny.hdr", "pz.hdr", "nz.hdr"];

        let mut cubemap = None;
        let mut resolution: u32 = 0;

        let mut cache_file = File::create(get_cache_filename(path)).unwrap();

        for (i, filename) in filenames.iter().enumerate() {
            info!("loading cubemap face {}..", i);

            let file_reader = std::io::BufReader::new(File::open(path.join(filename))?);
            let decoder = image::hdr::HdrDecoder::new(file_reader).unwrap();
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
                        depth_or_array_layers: 6,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: CUBEMAP_FORMAT,
                    usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
                }));
            }

            if resolution != metadata.width {
                panic!("all cubemap faces need to have the same resolution");
            }

            let image_data = decoder.read_image_native().unwrap();
            let image_data_raw =
                unsafe { std::slice::from_raw_parts(image_data.as_ptr() as *const u8, image_data.len() * std::mem::size_of::<Rgbe8Pixel>()) };
            cache_file.write_all(image_data_raw)?;

            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &cubemap.as_ref().unwrap(),
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: i as u32 },
                },
                image_data_raw,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: NonZeroU32::new(CUBEMAP_FORMAT_BYTES_PER_PIXEL * resolution),
                    rows_per_image: NonZeroU32::new(resolution),
                },
                wgpu::Extent3d {
                    width: resolution,
                    height: resolution,
                    depth_or_array_layers: 1,
                },
            );
        }

        Ok(cubemap.unwrap())
    }

    pub fn load(path: &Path, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<wgpu::TextureView, std::io::Error> {
        // Loading .hdr is somewhat slow, especially so in debug. So we cache the raw data.
        let cubemap = match from_cache(path, device, queue) {
            Ok(cubemap) => cubemap,
            Err(_) => {
                info!("no raw cubemap file, loading from .hdr faces instead");
                from_hdr_faces(path, device, queue)?
            }
        };

        Ok(cubemap.create_view(&wgpu::TextureViewDescriptor {
            label: None,
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..wgpu::TextureViewDescriptor::default()
        }))
    }
}

impl Background {
    pub fn new(
        path: &Path,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        shader_dir: &ShaderDirectory,
        pipeline_manager: &mut PipelineManager,
        global_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Result<Self, io::Error> {
        let file = File::open(path.join("config.json"))?;
        let reader = BufReader::new(file);
        let config: BackgroundConfig = serde_json::from_reader(reader)?;

        let ubo = LightingAndBackgroundUniformBuffer::new_with_data(
            &device,
            &LightingAndBackgroundUniformBufferContent {
                dir_light_direction: config.dir_light_direction.into(),
                dir_light_radiance: config.dir_light_radiance.into(),
                indirect_lighting_sh: [
                    (config.indirect_lighting_sh[0], 0.0),
                    (config.indirect_lighting_sh[1], 0.0),
                    (config.indirect_lighting_sh[2], 0.0),
                    (config.indirect_lighting_sh[3], 0.0),
                    (config.indirect_lighting_sh[4], 0.0),
                    (config.indirect_lighting_sh[5], 0.0),
                    (config.indirect_lighting_sh[6], 0.0),
                    (config.indirect_lighting_sh[7], 0.0),
                    (config.indirect_lighting_sh[8], 0.0),
                ],
            },
        );

        let cubemap_view = cubemap_loader::load(path, device, queue)?;

        let bind_group_layout = BindGroupLayoutBuilder::new()
            .next_binding(wgpu::ShaderStage::COMPUTE | wgpu::ShaderStage::FRAGMENT, binding_glsl::uniform())
            .next_binding(wgpu::ShaderStage::COMPUTE | wgpu::ShaderStage::FRAGMENT, binding_glsl::textureCube())
            .create(device, "BindGroupLayout: Lighting & Background");

        let bind_group = BindGroupBuilder::new(&bind_group_layout)
            .resource(ubo.binding_resource())
            .texture(&cubemap_view)
            .create(device, "BindGroup: Lighting & Background");

        let mut render_pipeline_desc = RenderPipelineCreationDesc::new(
            "Cubemap Renderer",
            Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Cubemap Renderer Pipeline Layout"),
                bind_group_layouts: &[&global_bind_group_layout, &bind_group_layout.layout],
                push_constant_ranges: &[],
            })),
            Path::new("screentri.vert"),
            Path::new("background_render.frag"),
            HdrBackbuffer::FORMAT,
            None,
        );
        render_pipeline_desc.depth_stencil = Some(wgpu::DepthStencilState {
            format: Screen::FORMAT_DEPTH,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: Default::default(),
            bias: Default::default(),
        });

        Ok(Background {
            pipeline: pipeline_manager.create_render_pipeline(device, shader_dir, render_pipeline_desc),
            bind_group_layout: bind_group_layout.layout,
            bind_group,
        })
    }

    pub fn draw<'a>(&'a self, rpass: &mut wgpu::RenderPass<'a>, pipeline_manager: &'a PipelineManager) {
        rpass.set_bind_group(1, &self.bind_group, &[]);
        rpass.set_pipeline(pipeline_manager.get_render(&self.pipeline));
        rpass.draw(0..3, 0..1);
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }
}
