use super::shader::ShaderDirectory;
use std::path::{Path, PathBuf};
use std::rc::Rc;

pub struct ReloadableComputePipeline {
    relative_shader_path: PathBuf,
    layout: Rc<wgpu::PipelineLayout>,
    pipeline: wgpu::ComputePipeline,
}

impl ReloadableComputePipeline {
    pub fn new(device: &wgpu::Device, layout: &Rc<wgpu::PipelineLayout>, shader_dir: &ShaderDirectory, relative_shader_path: &Path) -> Self {
        let pipeline = Self::try_create(device, layout, shader_dir, relative_shader_path).unwrap();

        ReloadableComputePipeline {
            relative_shader_path: PathBuf::from(relative_shader_path),
            layout: layout.clone(),
            pipeline,
        }
    }

    fn try_create(
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        shader_dir: &ShaderDirectory,
        relative_shader_path: &Path,
    ) -> Result<wgpu::ComputePipeline, ()> {
        let module = shader_dir.load_shader_module(device, relative_shader_path)?;
        Ok(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            layout,
            compute_stage: wgpu::ProgrammableStageDescriptor {
                module: &module,
                entry_point: super::shader::SHADER_ENTRY_POINT_NAME,
            },
        }))
    }

    pub fn try_reload_shader(&mut self, device: &wgpu::Device, shader_dir: &ShaderDirectory) -> Result<(), ()> {
        let pipeline = Self::try_create(device, &self.layout, shader_dir, &self.relative_shader_path)?;
        self.pipeline = pipeline;
        Ok(())
    }

    pub fn pipeline(&self) -> &wgpu::ComputePipeline {
        &self.pipeline
    }
}

pub mod rasterization_state {
    pub fn culling_none() -> wgpu::RasterizationStateDescriptor {
        wgpu::RasterizationStateDescriptor {
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: wgpu::CullMode::None,
            depth_bias: 0,
            depth_bias_slope_scale: 0.0,
            depth_bias_clamp: 0.0,
        }
    }
}

pub mod color_state {
    pub fn write_all(format: wgpu::TextureFormat) -> wgpu::ColorStateDescriptor {
        wgpu::ColorStateDescriptor {
            format: format,
            color_blend: wgpu::BlendDescriptor::REPLACE,
            alpha_blend: wgpu::BlendDescriptor::REPLACE,
            write_mask: wgpu::ColorWrite::ALL,
        }
    }
}

pub mod depth_state {
    pub fn default_read_write(format: wgpu::TextureFormat) -> wgpu::DepthStencilStateDescriptor {
        wgpu::DepthStencilStateDescriptor {
            format,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil_front: wgpu::StencilStateFaceDescriptor::IGNORE,
            stencil_back: wgpu::StencilStateFaceDescriptor::IGNORE,
            stencil_read_mask: 0,
            stencil_write_mask: 0,
        }
    }
}
