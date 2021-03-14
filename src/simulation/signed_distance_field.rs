use futures::FutureExt;
use std::{io::Read, io::Write, path::Path, rc::Rc};

use crate::{
    utils::round_to_multiple,
    wgpu_utils::{
        self,
        binding_builder::{BindGroupBuilder, BindGroupLayoutBuilder, BindGroupLayoutWithDesc},
        binding_glsl,
        pipelines::*,
        shader::ShaderDirectory,
    },
};

// TODO: Idea: The distance field needed for fluid sim isn't really signed, we're only interested in the negative part!

const DISTANCE_FIELD_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;
const DISTANCE_FIELD_BYTES_PER_VOXEL: u32 = 2;

pub struct SignedDistanceFieldBlockBuilder {
    pipeline_compute_distance_field: ComputePipelineHandle,
    group_layout_accumulate_signed_distance_field: BindGroupLayoutWithDesc,
}

impl SignedDistanceFieldBlockBuilder {
    pub fn new(
        device: &wgpu::Device,
        shader_dir: &ShaderDirectory,
        pipeline_manager: &mut PipelineManager,
        global_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let group_layout_accumulate_signed_distance_field = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::image3D(DISTANCE_FIELD_FORMAT, wgpu::StorageTextureAccess::ReadWrite))
            .create(device, "BindGroupLayout: Signed Distance Field Write");

        let layout_compute_distance_field = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PipelineLayout: Compute Distance Field"),
            bind_group_layouts: &[global_bind_group_layout, &group_layout_accumulate_signed_distance_field.layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStage::COMPUTE,
                range: 0..8,
            }],
        }));

        SignedDistanceFieldBlockBuilder {
            pipeline_compute_distance_field: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(
                    "Signed Distance Field Block from Mesh",
                    layout_compute_distance_field.clone(),
                    Path::new("simulation/distance_field_compute_block.comp"),
                ),
            ),
            group_layout_accumulate_signed_distance_field,
        }
    }

    const COMPUTE_LOCAL_SIZE: wgpu::Extent3d = wgpu::Extent3d {
        width: 4,
        height: 4,
        depth: 4,
    };

    pub fn compute_distance_field(
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
                    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("compute distance field"),
                    });
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
}

pub struct SignedDistanceFieldBlock {
    dimension: wgpu::Extent3d,
    volume: wgpu::Texture,
    volume_view: wgpu::TextureView,
}

impl SignedDistanceFieldBlock {
    fn new(device: &wgpu::Device, dimension: wgpu::Extent3d) -> Self {
        let volume = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Distance Field Block"),
            size: dimension,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: DISTANCE_FIELD_FORMAT,
            // TODO: For blocks from file we only need (wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST) so that could be faster
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::STORAGE | wgpu::TextureUsage::COPY_DST | wgpu::TextureUsage::COPY_SRC,
        });
        let volume_view = volume.create_view(&Default::default());

        SignedDistanceFieldBlock {
            dimension,
            volume,
            volume_view,
        }
    }

    pub fn load_from_file(&self, path: &Path, queue: &wgpu::Queue) -> Result<(), std::io::Error> {
        let mut raw_data = Vec::new();
        let num_bytes_read = std::fs::File::open(path)?.read_to_end(&mut raw_data).unwrap() as u32;
        if num_bytes_read != self.size_in_bytes() {
            error!(
                "Failure loading distance field from file {:?}: File size is {}, expected {}",
                path,
                num_bytes_read,
                self.size_in_bytes()
            );
            return Err(std::io::Error::new(std::io::ErrorKind::Other, "unexpected size"));
        }

        queue.write_texture(
            wgpu::TextureCopyView {
                texture: &self.volume,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            &raw_data,
            wgpu::TextureDataLayout {
                offset: 0,
                bytes_per_row: self.buffer_bytes_per_padded_row(),
                rows_per_image: self.dimension.height,
            },
            self.dimension,
        );

        Ok(())
    }

    // fairly brute force and blocking but we don't care here :)
    pub fn save(&self, path: &Path, device: &wgpu::Device, queue: &wgpu::Queue) {
        info!("Saving signed distance field data to {:?}", path);

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Signed Distance Field save temp buffer"),
            size: self.size_in_bytes() as wgpu::BufferAddress,
            usage: wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Encoder: Save Signed Distance Field"),
        });

        encoder.copy_texture_to_buffer(
            wgpu::TextureCopyView {
                texture: &self.volume,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::BufferCopyView {
                buffer: &buffer,
                layout: wgpu::TextureDataLayout {
                    offset: 0,
                    bytes_per_row: self.buffer_bytes_per_padded_row(),
                    rows_per_image: self.dimension.height,
                },
            },
            self.dimension,
        );
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        let buffer_slice = buffer.slice(..);
        let mapping = buffer_slice.map_async(wgpu::MapMode::Read);
        device.poll(wgpu::Maintain::Wait);
        mapping
            .now_or_never()
            .expect("Failed to await buffer mapping with signed distance field copy (Future not ready)")
            .expect("Failed to map buffer with signed distance field copy");

        let mut file = std::fs::File::create(path).expect(&format!("Failed to create file {:?}", path));
        file.write_all(&buffer_slice.get_mapped_range())
            .expect("Failed to write signed distance field data");
    }

    fn size_in_bytes(&self) -> u32 {
        self.buffer_bytes_per_padded_row() * self.dimension.height * self.dimension.depth
    }

    fn buffer_bytes_per_padded_row(&self) -> u32 {
        round_to_multiple(
            self.dimension.width as usize * DISTANCE_FIELD_BYTES_PER_VOXEL as usize,
            wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize,
        ) as u32
    }

    pub fn texture_view(&self) -> &wgpu::TextureView {
        &self.volume_view
    }
}

pub struct CombinedSignedDistanceField {
    pipeline_distance_field_accumulate: ComputePipelineHandle,
    volume: SignedDistanceFieldBlock,
}

impl CombinedSignedDistanceField {
    pub fn new(
        device: &wgpu::Device,
        grid_dimension: wgpu::Extent3d,
        shader_dir: &ShaderDirectory,
        pipeline_manager: &mut PipelineManager,
        global_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let group_layout_accumulate_signed_distance_field = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::image3D(
                wgpu::TextureFormat::R16Float,
                wgpu::StorageTextureAccess::ReadWrite,
            ))
            .create(device, "BindGroupLayout: Signed Distance Field Accumulate");

        let volume = SignedDistanceFieldBlock::new(device, grid_dimension);

        let bind_group_accumulate_signed_distance_field = BindGroupBuilder::new(&group_layout_accumulate_signed_distance_field)
            .texture(&volume.texture_view())
            .create(device, "BindGroup: Write Distance Field");

        let layout_compute_distance_field = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PipelineLayout: Compute Distance Field"),
            bind_group_layouts: &[global_bind_group_layout, &group_layout_accumulate_signed_distance_field.layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStage::COMPUTE,
                range: 0..8,
            }],
        }));

        CombinedSignedDistanceField {
            volume,
            pipeline_distance_field_accumulate: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(
                    "Distance Field accumulate",
                    layout_compute_distance_field.clone(),
                    Path::new("simulation/distance_field_accumulate.comp"),
                ),
            ),
        }
    }

    pub fn clear(&self) {
        // Set some flag so that the first accumulated
    }

    pub fn accumulate(&self, block: &SignedDistanceFieldBlock) {}
}
