use super::shader::{ShaderDirectory, SHADER_ENTRY_POINT_NAME};
use std::path::{Path, PathBuf};
use std::rc::{Rc, Weak};

pub type ComputePipelineHandle = Rc<usize>;
pub type RenderPipelineHandle = Rc<usize>;

// This is essentially a copy of wgpu::ComputePipelineDescriptor that we can store.
// This is needed since we want to be able to reload pipelines while the program is running.
pub struct ComputePipelineCreationDesc {
    /// Debug label of the pipeline. This will show up in graphics debuggers for easy identification.
    pub label: &'static str,
    /// The layout of bind groups for this pipeline.
    pub layout: Rc<wgpu::PipelineLayout>,
    pub compute_shader_relative_path: PathBuf,
}

impl ComputePipelineCreationDesc {
    pub fn new(label: &'static str, layout: Rc<wgpu::PipelineLayout>, compute_shader_relative_path: &Path) -> Self {
        ComputePipelineCreationDesc {
            label,
            layout,
            compute_shader_relative_path: PathBuf::from(compute_shader_relative_path),
        }
    }

    fn try_create_pipeline(&self, device: &wgpu::Device, shader_dir: &ShaderDirectory) -> Result<PipelineAndSourceFiles<wgpu::ComputePipeline>, ()> {
        let shader = shader_dir.load_shader_module(device, &self.compute_shader_relative_path)?;
        Ok(PipelineAndSourceFiles {
            pipeline: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(self.label),
                layout: Some(&self.layout),
                module: &shader.module,
                entry_point: SHADER_ENTRY_POINT_NAME,
            }),
            shader_sources: shader.source_files,
        })
    }
}

pub struct VertexStateCreationDesc {
    /// Path to shader source file.
    pub shader_relative_path: PathBuf,
    /// The format of any vertex buffers used with this pipeline.
    pub buffers: Vec<wgpu::VertexBufferLayout<'static>>,
}

/// Describes the fragment process in a render pipeline.
#[derive(Clone, Debug)]
pub struct FragmentStateCreationDesc {
    /// Path to shader source file.
    pub shader_relative_path: PathBuf,
    /// The format of any vertex buffers used with this pipeline.
    pub targets: Vec<wgpu::ColorTargetState>,
}

// This is essentially a copy of wgpu::RenderPipelineDescriptor that we can store.
// This is needed since we want to be able to reload pipelines while the program is running.
pub struct RenderPipelineCreationDesc {
    /// Debug label of the pipeline. This will show up in graphics debuggers for easy identification.
    pub label: &'static str,
    /// The layout of bind groups for this pipeline.
    pub layout: Rc<wgpu::PipelineLayout>,
    /// The vertex stage, its entry point, and the input buffers layout.
    pub vertex: VertexStateCreationDesc,
    /// The properties of the pipeline at the primitive assembly and rasterization level.
    pub primitive: wgpu::PrimitiveState,
    /// The effect of draw calls on the depth and stencil aspects of the output target, if any.
    pub depth_stencil: Option<wgpu::DepthStencilState>,
    /// The multi-sampling properties of the pipeline.
    pub multisample: wgpu::MultisampleState,
    /// The fragment stage, its entry point, and the color targets.
    pub fragment: FragmentStateCreationDesc,
}

impl RenderPipelineCreationDesc {
    pub fn new(
        label: &'static str,
        layout: Rc<wgpu::PipelineLayout>,
        vertex_shader_relative_path: &Path,
        fragment_shader_relative_path: &Path,
        output_format: wgpu::TextureFormat,
        depth_format: Option<wgpu::TextureFormat>,
    ) -> Self {
        RenderPipelineCreationDesc {
            label,
            layout,
            vertex: VertexStateCreationDesc {
                shader_relative_path: PathBuf::from(vertex_shader_relative_path),
                buffers: Vec::new(),
            },
            primitive: Default::default(),
            depth_stencil: match depth_format {
                Some(depth_format) => Some(depth_state::default_read_write(depth_format)),
                None => None,
            },
            multisample: Default::default(),
            fragment: FragmentStateCreationDesc {
                shader_relative_path: PathBuf::from(fragment_shader_relative_path),
                targets: vec![output_format.into()],
            },
        }
    }

    fn try_create_pipeline(&self, device: &wgpu::Device, shader_dir: &ShaderDirectory) -> Result<PipelineAndSourceFiles<wgpu::RenderPipeline>, ()> {
        let shader_vs = shader_dir.load_shader_module(device, &self.vertex.shader_relative_path)?;
        let mut shader_fs = shader_dir.load_shader_module(device, &self.fragment.shader_relative_path)?;

        let render_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some(self.label),
            layout: Some(&self.layout),
            vertex: wgpu::VertexState {
                module: &shader_vs.module,
                entry_point: SHADER_ENTRY_POINT_NAME,
                buffers: &self.vertex.buffers,
            },
            primitive: self.primitive.clone(),
            depth_stencil: self.depth_stencil.clone(),
            multisample: self.multisample.clone(),
            fragment: Some(wgpu::FragmentState {
                module: &shader_fs.module,
                entry_point: SHADER_ENTRY_POINT_NAME,
                targets: &self.fragment.targets,
            }),
        };

        let mut shader_sources = shader_vs.source_files;
        shader_sources.append(&mut shader_fs.source_files);

        Ok(PipelineAndSourceFiles {
            pipeline: device.create_render_pipeline(&render_pipeline_descriptor),
            shader_sources,
        })
    }
}

struct PipelineAndSourceFiles<T> {
    pipeline: T,
    shader_sources: Vec<PathBuf>,
}

struct ReloadableComputePipeline {
    desc: ComputePipelineCreationDesc,
    handle: Weak<usize>,
    pipeline_and_sources: PipelineAndSourceFiles<wgpu::ComputePipeline>,
}

struct ReloadableRenderPipeline {
    desc: RenderPipelineCreationDesc,
    handle: Weak<usize>,
    pipeline_and_sources: PipelineAndSourceFiles<wgpu::RenderPipeline>,
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
        let pipeline_and_sources = desc.try_create_pipeline(device, shader_dir).unwrap();

        let mut first_free_slot = 0;
        while first_free_slot < self.compute_pipelines.len() && self.compute_pipelines[first_free_slot].handle.strong_count() > 0 {
            first_free_slot += 1;
        }

        let handle = Rc::new(first_free_slot);
        let new_reloadable_pipeline = ReloadableComputePipeline {
            desc,
            pipeline_and_sources,
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
        let pipeline_and_sources = desc.try_create_pipeline(device, shader_dir).unwrap();

        let mut first_free_slot = 0;
        while first_free_slot < self.render_pipelines.len() && self.render_pipelines[first_free_slot].handle.strong_count() > 0 {
            first_free_slot += 1;
        }

        let handle = Rc::new(first_free_slot);
        let new_reloadable_pipeline = ReloadableRenderPipeline {
            desc,
            pipeline_and_sources,
            handle: Rc::downgrade(&handle),
        };
        if first_free_slot == self.render_pipelines.len() {
            self.render_pipelines.push(new_reloadable_pipeline);
        } else {
            self.render_pipelines[first_free_slot] = new_reloadable_pipeline;
        }
        handle
    }

    pub fn reload_changed(&mut self, device: &wgpu::Device, shader_dir: &ShaderDirectory, changed_shader_sources: &[PathBuf]) {
        for reloadable_pipeline in self.compute_pipelines.iter_mut() {
            if changed_shader_sources.iter().any(|p| {
                if reloadable_pipeline.pipeline_and_sources.shader_sources.contains(p) {
                    info!(
                        "Reloading compute pipeline \"{}\" because {:?} changed",
                        reloadable_pipeline.desc.label, p
                    );
                    true
                } else {
                    false
                }
            }) {
                if let Ok(pipeline_and_sources) = reloadable_pipeline.desc.try_create_pipeline(device, shader_dir) {
                    reloadable_pipeline.pipeline_and_sources = pipeline_and_sources;
                }
            }
        }
        for reloadable_pipeline in self.render_pipelines.iter_mut() {
            if changed_shader_sources.iter().any(|p| {
                if reloadable_pipeline.pipeline_and_sources.shader_sources.contains(p) {
                    info!("Reloading render pipeline \"{}\" because {:?} changed", reloadable_pipeline.desc.label, p);
                    true
                } else {
                    false
                }
            }) {
                if let Ok(pipeline_and_sources) = reloadable_pipeline.desc.try_create_pipeline(device, shader_dir) {
                    reloadable_pipeline.pipeline_and_sources = pipeline_and_sources;
                }
            }
        }
    }

    pub fn get_compute(&self, handle: &ComputePipelineHandle) -> &wgpu::ComputePipeline {
        let i: usize = **handle;
        assert!(self.compute_pipelines[i].handle.ptr_eq(&Rc::downgrade(handle)));
        &self.compute_pipelines[i].pipeline_and_sources.pipeline
    }

    pub fn get_render(&self, handle: &RenderPipelineHandle) -> &wgpu::RenderPipeline {
        let i: usize = **handle;
        assert!(self.render_pipelines[i].handle.ptr_eq(&Rc::downgrade(handle)));
        &self.render_pipelines[i].pipeline_and_sources.pipeline
    }
}

pub mod depth_state {
    pub fn default_read_write(format: wgpu::TextureFormat) -> wgpu::DepthStencilState {
        wgpu::DepthStencilState {
            format,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: Default::default(),
            bias: Default::default(),
        }
    }
}
