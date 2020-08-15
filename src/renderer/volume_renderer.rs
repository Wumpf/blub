use crate::shader::ShaderDirectory;
use crate::{
    render_output::{hdr_backbuffer::HdrBackbuffer, screen::Screen},
    simulation::HybridFluid,
    wgpu_utils::pipelines::*,
};
use std::{path::Path, rc::Rc};

#[derive(Clone, Copy, Debug, EnumIter)]
pub enum VolumeVisualizationMode {
    None,
    Velocity,
    DivergenceError,
    PseudoPressure,
    UncorrectedDensityError,
    Marker,
}

pub struct VolumeRenderer {
    velocity_render_pipeline: RenderPipelineHandle,
    volume_visualization_with_billboards_pipeline: RenderPipelineHandle,
}

impl VolumeRenderer {
    pub fn new(
        device: &wgpu::Device,
        shader_dir: &ShaderDirectory,
        pipeline_manager: &mut PipelineManager,
        per_frame_bind_group_layout: &wgpu::BindGroupLayout,
        fluid_renderer_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let layout = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Volume Renderer Pipeline Layout"),
            bind_group_layouts: &[&per_frame_bind_group_layout, &fluid_renderer_group_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStage::VERTEX,
                range: 0..4,
            }],
        }));

        let mut velocity_render_pipeline_desc = RenderPipelineCreationDesc::new(
            layout.clone(),
            Path::new("volume_visualization/velocity.vert"),
            Some(Path::new("vertex_color.frag")),
            HdrBackbuffer::FORMAT,
            Some(Screen::FORMAT_DEPTH),
        );
        velocity_render_pipeline_desc.primitive_topology = wgpu::PrimitiveTopology::LineList;

        let volume_visualization_with_billboards_pipeline_desc = RenderPipelineCreationDesc::new(
            layout.clone(),
            Path::new("volume_visualization/volume_visualization_with_billboards.vert"),
            Some(Path::new("sphere_particles.frag")),
            HdrBackbuffer::FORMAT,
            Some(Screen::FORMAT_DEPTH),
        );

        VolumeRenderer {
            velocity_render_pipeline: pipeline_manager.create_render_pipeline(device, shader_dir, velocity_render_pipeline_desc),
            volume_visualization_with_billboards_pipeline: pipeline_manager.create_render_pipeline(
                device,
                shader_dir,
                volume_visualization_with_billboards_pipeline_desc,
            ),
        }
    }

    fn num_grid_cells(dimension: wgpu::Extent3d) -> u32 {
        dimension.width * dimension.height * dimension.depth
    }

    pub fn draw<'a>(
        &'a self,
        rpass: &mut wgpu::RenderPass<'a>,
        pipeline_manager: &'a PipelineManager,
        fluid: &'a HybridFluid,
        mode: VolumeVisualizationMode,
    ) {
        match mode {
            VolumeVisualizationMode::None => {}
            VolumeVisualizationMode::Velocity => {
                rpass.set_pipeline(pipeline_manager.get_render(&self.velocity_render_pipeline));
                rpass.set_bind_group(1, fluid.bind_group_renderer(), &[]);
                rpass.draw(0..2, 0..Self::num_grid_cells(fluid.grid_dimension()) * 3);
            }
            _ => {
                rpass.set_pipeline(pipeline_manager.get_render(&self.volume_visualization_with_billboards_pipeline));
                rpass.set_bind_group(1, fluid.bind_group_renderer(), &[]);
                match mode {
                    VolumeVisualizationMode::DivergenceError => rpass.set_push_constants(wgpu::ShaderStage::VERTEX, 0, &[0]),
                    VolumeVisualizationMode::PseudoPressure => rpass.set_push_constants(wgpu::ShaderStage::VERTEX, 0, &[1]),
                    VolumeVisualizationMode::UncorrectedDensityError => rpass.set_push_constants(wgpu::ShaderStage::VERTEX, 0, &[2]),
                    VolumeVisualizationMode::Marker => rpass.set_push_constants(wgpu::ShaderStage::VERTEX, 0, &[3]),
                    _ => {}
                };
                rpass.draw(0..6, 0..Self::num_grid_cells(fluid.grid_dimension()));
            }
        }
    }
}
