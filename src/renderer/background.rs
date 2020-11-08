use crate::{
    render_output::hdr_backbuffer::HdrBackbuffer,
    render_output::screen::Screen,
    wgpu_utils::uniformbuffer::PaddedVector3,
    wgpu_utils::{binding_builder::*, binding_glsl, pipelines::*, shader::ShaderDirectory, uniformbuffer::UniformBuffer},
};
use image::hdr::{HdrDecoder, Rgbe8Pixel};
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

// Loads cubemap in rgbe format
fn load_cubemap(path: &Path, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<wgpu::TextureView, io::Error> {
    let filenames = ["px.hdr", "nx.hdr", "py.hdr", "ny.hdr", "pz.hdr", "nz.hdr"];

    let mut cubemap = None;
    let mut resolution: u32 = 0;

    for (i, filename) in filenames.iter().enumerate() {
        info!("loading cubemap face {}..", i);

        let file_reader = BufReader::new(File::open(path.join(filename))?);
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

    Ok(cubemap.unwrap().create_view(&wgpu::TextureViewDescriptor {
        label: None,
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..wgpu::TextureViewDescriptor::default()
    }))
}

impl Background {
    pub fn new(
        path: &Path,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        shader_dir: &ShaderDirectory,
        pipeline_manager: &mut PipelineManager,
        per_frame_bind_group_layout: &wgpu::BindGroupLayout,
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

        let cubemap_view = load_cubemap(path, device, queue)?;

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
                bind_group_layouts: &[&per_frame_bind_group_layout, &bind_group_layout.layout],
                push_constant_ranges: &[],
            })),
            Path::new("screentri.vert"),
            Some(Path::new("background_render.frag")),
            HdrBackbuffer::FORMAT,
            None,
        );
        render_pipeline_desc.depth_stencil_state = Some(wgpu::DepthStencilStateDescriptor {
            format: Screen::FORMAT_DEPTH,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: Default::default(),
        });

        Ok(Background {
            pipeline: pipeline_manager.create_render_pipeline(device, shader_dir, render_pipeline_desc),
            bind_group_layout: bind_group_layout.layout,
            bind_group,
        })
    }

    pub fn draw<'a>(&'a self, rpass: &mut wgpu::RenderPass<'a>, pipeline_manager: &'a PipelineManager) {
        wgpu_scope!(rpass, "CubemapRenderer.draw");
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
