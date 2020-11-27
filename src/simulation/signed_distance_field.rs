use std::{path::Path, rc::Rc};

use crate::wgpu_utils::{
    self,
    binding_builder::{BindGroupBuilder, BindGroupLayoutBuilder},
    binding_glsl,
    pipelines::*,
    shader::ShaderDirectory,
};

pub struct SignedDistanceField {
    grid_dimension: wgpu::Extent3d,
    bind_group_write_signed_distance_field: wgpu::BindGroup,
    pipeline_compute_distance_field: ComputePipelineHandle,
    volume_signed_distances_view: wgpu::TextureView,
}

impl SignedDistanceField {
    pub fn new(
        device: &wgpu::Device,
        grid_dimension: wgpu::Extent3d,
        shader_dir: &ShaderDirectory,
        pipeline_manager: &mut PipelineManager,
        global_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let group_layout_write_signed_distance_field = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::image3D(wgpu::TextureFormat::R16Float, false))
            .create(device, "BindGroupLayout: Signed Distance Field Write");

        let volume_solid_signed_distances = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Signed Distance Field"),
            size: grid_dimension,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R16Float,
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::STORAGE,
        });
        let volume_signed_distances_view = volume_solid_signed_distances.create_view(&Default::default());

        let bind_group_write_signed_distance_field = BindGroupBuilder::new(&group_layout_write_signed_distance_field)
            .texture(&volume_signed_distances_view)
            .create(device, "BindGroup: Write Distance Field");

        let layout_compute_distance_field = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PipelineLayout: Compute Distance Field"),
            bind_group_layouts: &[global_bind_group_layout, &group_layout_write_signed_distance_field.layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStage::COMPUTE,
                range: 0..8,
            }],
        }));

        SignedDistanceField {
            grid_dimension,
            pipeline_compute_distance_field: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(
                    "Signed Distance Field from Mesh",
                    layout_compute_distance_field.clone(),
                    Path::new("simulation/compute_distance_field.comp"),
                ),
            ),
            bind_group_write_signed_distance_field,
            volume_signed_distances_view,
        }
    }

    // TODO: Smaller?
    const COMPUTE_LOCAL_SIZE: wgpu::Extent3d = wgpu::Extent3d {
        width: 4,
        height: 4,
        depth: 4,
    };

    pub fn compute_distance_field_for_static(
        &self,
        device: &wgpu::Device,
        pipeline_manager: &PipelineManager,
        queue: &wgpu::Queue,
        global_bind_group: &wgpu::BindGroup,
        meshes: &Vec<crate::scene_models::MeshData>,
    ) {
        // Brute force signed distance field computation.
        // Chunked up into several operations each with full wait so we don't run into TDR.
        info!("Static signed distance field is computed brute force on GPU...");
        let start_time_overall = std::time::Instant::now();

        let grid_work_groups = wgpu_utils::compute_group_size(self.grid_dimension, Self::COMPUTE_LOCAL_SIZE);

        const INDEX_CHUNK_SIZE: u32 = 16384 * 3; // max 16384 triangles at a time.
        for (mesh_idx, mesh) in meshes.iter().enumerate() {
            let mut start_index = mesh.index_buffer_range.start;
            while start_index < mesh.index_buffer_range.end {
                let end_index = std::cmp::min(mesh.index_buffer_range.end, start_index + INDEX_CHUNK_SIZE);
                let start_time = std::time::Instant::now();
                trace!("adding {} triangles...", (end_index - start_index) / 3);

                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Encoder: Distance Field Compute"),
                });

                {
                    let mut compute_pass = encoder.begin_compute_pass();
                    compute_pass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_compute_distance_field));
                    compute_pass.set_bind_group(0, global_bind_group, &[]);
                    compute_pass.set_bind_group(1, &self.bind_group_write_signed_distance_field, &[]);
                    compute_pass.set_push_constants(0, bytemuck::bytes_of(&[mesh_idx as u32, start_index]));
                    compute_pass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);
                }

                queue.submit(Some(encoder.finish()));
                device.poll(wgpu::Maintain::Wait);

                trace!("(took {:?})", start_time.elapsed());
                start_index += INDEX_CHUNK_SIZE;
            }
        }

        info!("Static signed distance field computation took {:?}", start_time_overall.elapsed());
    }

    pub fn texture_view(&self) -> &wgpu::TextureView {
        &self.volume_signed_distances_view
    }
}
