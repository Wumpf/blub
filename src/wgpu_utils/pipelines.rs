use super::shader::ShaderDirectory;
use std::path::{Path, PathBuf};
use std::rc::{Rc, Weak};

pub type PipelineHandle = Rc<usize>;

pub struct ComputePipelineDesc {
    pub layout: Rc<wgpu::PipelineLayout>,
    pub relative_shader_path: PathBuf,
}

pub struct ReloadableComputePipeline {
    desc: ComputePipelineDesc,
    pipeline: wgpu::ComputePipeline,
    handle: Weak<usize>,
}

pub struct PipelineManager {
    compute_pipelines: Vec<ReloadableComputePipeline>,
}

impl PipelineManager {
    pub fn new() -> Self {
        PipelineManager {
            compute_pipelines: Vec::new(),
        }
    }

    // TODO: Do also render pipelines!

    pub fn create_compute_pipeline(
        &mut self,
        device: &wgpu::Device,
        shader_dir: &ShaderDirectory,
        layout: &Rc<wgpu::PipelineLayout>,
        relative_shader_path: &Path,
    ) -> PipelineHandle {
        let desc = ComputePipelineDesc {
            layout: layout.clone(),
            relative_shader_path: PathBuf::from(relative_shader_path),
        };
        let pipeline = Self::try_create_compute(device, shader_dir, &desc).unwrap();

        let mut first_free_slot = 0;
        while first_free_slot < self.compute_pipelines.len() && self.compute_pipelines[first_free_slot].handle.strong_count() > 0 {
            first_free_slot += 1;
        }

        let handle = Rc::new(first_free_slot);
        let new_reloadable_pipeline = ReloadableComputePipeline {
            desc,
            pipeline,
            handle: Rc::downgrade(&handle),
        };
        if first_free_slot == self.compute_pipelines.len() {
            self.compute_pipelines.push(new_reloadable_pipeline);
        } else {
            self.compute_pipelines[first_free_slot] = new_reloadable_pipeline;
        }
        handle
    }

    // todo: reload only what's necessary
    pub fn reload_all(&mut self, device: &wgpu::Device, shader_dir: &ShaderDirectory) {
        for reloadable_pipeline in self.compute_pipelines.iter_mut() {
            if let Ok(new_wgpu_pipeline) = Self::try_create_compute(device, shader_dir, &reloadable_pipeline.desc) {
                reloadable_pipeline.pipeline = new_wgpu_pipeline;
            }
        }
    }

    fn try_create_compute(device: &wgpu::Device, shader_dir: &ShaderDirectory, desc: &ComputePipelineDesc) -> Result<wgpu::ComputePipeline, ()> {
        let module = shader_dir.load_shader_module(device, &desc.relative_shader_path)?;
        Ok(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            layout: &desc.layout,
            compute_stage: wgpu::ProgrammableStageDescriptor {
                module: &module,
                entry_point: super::shader::SHADER_ENTRY_POINT_NAME,
            },
        }))
    }

    pub fn get_compute(&self, index: &PipelineHandle) -> &wgpu::ComputePipeline {
        let i: usize = **index;
        assert!(self.compute_pipelines[i].handle.strong_count() > 0);
        assert!(*self.compute_pipelines[i].handle.upgrade().unwrap() == i);
        &self.compute_pipelines[i].pipeline
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
