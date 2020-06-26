use super::shader::{ShaderDirectory, SHADER_ENTRY_POINT_NAME};
use crate::render_output::screen::Screen;
use std::path::{Path, PathBuf};
use std::rc::{Rc, Weak};

pub type ComputePipelineHandle = Rc<usize>;
pub type RenderPipelineHandle = Rc<usize>;

pub struct ComputePipelineCreationDesc {
    pub layout: Rc<wgpu::PipelineLayout>,
    pub compute_shader_relative_path: PathBuf,
}

impl ComputePipelineCreationDesc {
    pub fn new(layout: Rc<wgpu::PipelineLayout>, compute_shader_relative_path: &Path) -> Self {
        ComputePipelineCreationDesc {
            layout,
            compute_shader_relative_path: PathBuf::from(compute_shader_relative_path),
        }
    }

    fn try_create_pipeline(&self, device: &wgpu::Device, shader_dir: &ShaderDirectory) -> Result<wgpu::ComputePipeline, ()> {
        let module = shader_dir.load_shader_module(device, &self.compute_shader_relative_path)?;
        Ok(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            layout: &self.layout,
            compute_stage: wgpu::ProgrammableStageDescriptor {
                module: &module,
                entry_point: super::shader::SHADER_ENTRY_POINT_NAME,
            },
        }))
    }
}

pub struct RenderPipelineCreationDesc {
    /// The layout of bind groups for this pipeline.
    pub layout: Rc<wgpu::PipelineLayout>,

    /// The compiled vertex stage and its entry point.
    pub vertex_shader_relative_path: PathBuf,

    /// The compiled fragment stage and its entry point, if any.
    pub fragment_shader_relative_path: Option<PathBuf>,

    /// The rasterization process for this pipeline.
    pub rasterization_state: Option<wgpu::RasterizationStateDescriptor>,

    /// The primitive topology used to interpret vertices.
    pub primitive_topology: wgpu::PrimitiveTopology,

    /// The effect of draw calls on the color aspect of the output target.
    pub color_states: Vec<wgpu::ColorStateDescriptor>,

    /// The effect of draw calls on the depth and stencil aspects of the output target, if any.
    pub depth_stencil_state: Option<wgpu::DepthStencilStateDescriptor>,

    /// The vertex input state for this pipeline.
    pub vertex_state: wgpu::VertexStateDescriptor<'static>,

    /// The number of samples calculated per pixel (for MSAA).
    pub sample_count: u32,

    /// Bitmask that restricts the samples of a pixel modified by this pipeline.
    pub sample_mask: u32,

    /// When enabled, produces another sample mask per pixel based on the alpha output value, that
    /// is ANDed with the sample_mask and the primitive coverage to restrict the set of samples
    /// affected by a primitive.
    /// The implicit mask produced for alpha of zero is guaranteed to be zero, and for alpha of one
    /// is guaranteed to be all 1-s.
    pub alpha_to_coverage_enabled: bool,
}

impl RenderPipelineCreationDesc {
    pub fn new(layout: Rc<wgpu::PipelineLayout>, vertex_shader_relative_path: &Path, fragment_shader_relative_path: Option<&Path>) -> Self {
        RenderPipelineCreationDesc {
            layout: layout,
            vertex_shader_relative_path: PathBuf::from(vertex_shader_relative_path),
            fragment_shader_relative_path: match fragment_shader_relative_path {
                None => None,
                Some(path) => Some(PathBuf::from(path)),
            },
            rasterization_state: Some(rasterization_state::culling_none()), // culling none is a curious default...
            primitive_topology: wgpu::PrimitiveTopology::TriangleStrip,
            color_states: vec![color_state::write_all(Screen::FORMAT_BACKBUFFER)],
            depth_stencil_state: Some(depth_state::default_read_write(Screen::FORMAT_DEPTH)),
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: wgpu::IndexFormat::Uint16,
                vertex_buffers: &[],
            },
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        }
    }

    fn try_create_pipeline(&self, device: &wgpu::Device, shader_dir: &ShaderDirectory) -> Result<wgpu::RenderPipeline, ()> {
        let vs_module = shader_dir.load_shader_module(device, &self.vertex_shader_relative_path)?;
        let fs_module = match &self.fragment_shader_relative_path {
            None => None,
            Some(relative_path) => Some(shader_dir.load_shader_module(device, relative_path)?),
        };

        let render_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            layout: &self.layout,
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &vs_module,
                entry_point: SHADER_ENTRY_POINT_NAME,
            },
            fragment_stage: match &fs_module {
                None => None,
                Some(module) => Some(wgpu::ProgrammableStageDescriptor {
                    module,
                    entry_point: SHADER_ENTRY_POINT_NAME,
                }),
            },
            rasterization_state: self.rasterization_state.clone(),
            primitive_topology: self.primitive_topology,
            color_states: &self.color_states,
            depth_stencil_state: self.depth_stencil_state.clone(),
            vertex_state: self.vertex_state.clone(),
            sample_count: self.sample_count,
            sample_mask: self.sample_mask,
            alpha_to_coverage_enabled: self.alpha_to_coverage_enabled,
        };

        Ok(device.create_render_pipeline(&render_pipeline_descriptor))
    }
}

struct ReloadableComputePipeline {
    desc: ComputePipelineCreationDesc,
    pipeline: wgpu::ComputePipeline,
    handle: Weak<usize>,
}

struct ReloadableRenderPipeline {
    desc: RenderPipelineCreationDesc,
    pipeline: wgpu::RenderPipeline,
    handle: Weak<usize>,
}

pub struct PipelineManager {
    compute_pipelines: Vec<ReloadableComputePipeline>,
    render_pipelines: Vec<ReloadableRenderPipeline>,
}

impl PipelineManager {
    pub fn new() -> Self {
        PipelineManager {
            compute_pipelines: Vec::new(),
            render_pipelines: Vec::new(),
        }
    }

    pub fn create_compute_pipeline(
        &mut self,
        device: &wgpu::Device,
        shader_dir: &ShaderDirectory,
        desc: ComputePipelineCreationDesc,
    ) -> ComputePipelineHandle {
        let pipeline = desc.try_create_pipeline(device, shader_dir).unwrap();

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

    pub fn create_render_pipeline(
        &mut self,
        device: &wgpu::Device,
        shader_dir: &ShaderDirectory,
        desc: RenderPipelineCreationDesc,
    ) -> RenderPipelineHandle {
        // duplicated code from create_compute_pipeline, but well within Rule of Three ;-)
        let pipeline = desc.try_create_pipeline(device, shader_dir).unwrap();

        let mut first_free_slot = 0;
        while first_free_slot < self.render_pipelines.len() && self.render_pipelines[first_free_slot].handle.strong_count() > 0 {
            first_free_slot += 1;
        }

        let handle = Rc::new(first_free_slot);
        let new_reloadable_pipeline = ReloadableRenderPipeline {
            desc,
            pipeline,
            handle: Rc::downgrade(&handle),
        };
        if first_free_slot == self.render_pipelines.len() {
            self.render_pipelines.push(new_reloadable_pipeline);
        } else {
            self.render_pipelines[first_free_slot] = new_reloadable_pipeline;
        }
        handle
    }

    // todo: reload only what's necessary
    pub fn reload_all(&mut self, device: &wgpu::Device, shader_dir: &ShaderDirectory) {
        for reloadable_pipeline in self.compute_pipelines.iter_mut() {
            if let Ok(new_wgpu_pipeline) = reloadable_pipeline.desc.try_create_pipeline(device, shader_dir) {
                reloadable_pipeline.pipeline = new_wgpu_pipeline;
            }
        }
        for reloadable_pipeline in self.render_pipelines.iter_mut() {
            if let Ok(new_wgpu_pipeline) = reloadable_pipeline.desc.try_create_pipeline(device, shader_dir) {
                reloadable_pipeline.pipeline = new_wgpu_pipeline;
            }
        }
    }

    pub fn get_compute(&self, handle: &ComputePipelineHandle) -> &wgpu::ComputePipeline {
        let i: usize = **handle;
        assert!(self.compute_pipelines[i].handle.ptr_eq(&Rc::downgrade(handle)));
        &self.compute_pipelines[i].pipeline
    }

    pub fn get_render(&self, handle: &RenderPipelineHandle) -> &wgpu::RenderPipeline {
        let i: usize = **handle;
        assert!(self.render_pipelines[i].handle.ptr_eq(&Rc::downgrade(handle)));
        &self.render_pipelines[i].pipeline
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
