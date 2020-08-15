use crate::wgpu_utils;
use crate::wgpu_utils::binding_builder::*;
use crate::wgpu_utils::binding_glsl;
use crate::wgpu_utils::pipelines::*;
use crate::wgpu_utils::shader::*;
use crate::wgpu_utils::uniformbuffer::*;
use rand::prelude::*;
use std::{cell::Cell, path::Path, rc::Rc};

#[repr(C)]
#[derive(Clone, Copy)]
struct SimulationPropertiesUniformBufferContent {
    gravity_grid: cgmath::Vector3<f32>,
    num_particles: u32,
}
unsafe impl bytemuck::Pod for SimulationPropertiesUniformBufferContent {}
unsafe impl bytemuck::Zeroable for SimulationPropertiesUniformBufferContent {}

pub struct HybridFluid {
    grid_dimension: wgpu::Extent3d,

    particles_position_llindex: wgpu::Buffer,
    particles_velocity_x: wgpu::Buffer,
    particles_velocity_y: wgpu::Buffer,
    particles_velocity_z: wgpu::Buffer,
    simulation_properties_uniformbuffer: UniformBuffer<SimulationPropertiesUniformBufferContent>,
    simulation_properties: SimulationPropertiesUniformBufferContent,
    simulation_properties_dirty: Cell<bool>,

    bind_group_uniform: wgpu::BindGroup,

    bind_group_transfer_velocity: [wgpu::BindGroup; 3],
    bind_group_write_velocity: wgpu::BindGroup,

    bind_group_read_mac_grid: wgpu::BindGroup,
    bind_group_pressure_compute_divergence: wgpu::BindGroup,
    bind_group_pressure_preconditioner: [wgpu::BindGroup; 3],
    bind_group_pressure_apply_coeff: wgpu::BindGroup,
    bind_group_pressure_dotproduct_reduce: [wgpu::BindGroup; 2],
    bind_group_pressure_dotproduct_final: [wgpu::BindGroup; 2],
    bind_group_pressure_update_pressure_and_residual: wgpu::BindGroup,
    bind_group_pressure_update_search: wgpu::BindGroup,

    bind_group_advect_particles: wgpu::BindGroup,

    bind_group_density_projection_gather: wgpu::BindGroup,

    // The interface to any renderer of the fluid. Readonly access to relevant resources
    bind_group_renderer: wgpu::BindGroup,

    pipeline_transfer_clear: ComputePipelineHandle,
    pipeline_transfer_build_linkedlist: ComputePipelineHandle,
    pipeline_transfer_gather_velocity: ComputePipelineHandle,

    pipeline_pressure_compute_divergence: ComputePipelineHandle,
    pipeline_pressure_apply_preconditioner: ComputePipelineHandle,
    pipeline_pressure_reduce: ComputePipelineHandle,
    pipeline_pressure_apply_coeff: ComputePipelineHandle,
    pipeline_pressure_update_pressure_and_residual: ComputePipelineHandle,
    pipeline_pressure_update_search: ComputePipelineHandle,

    pipeline_remove_divergence: ComputePipelineHandle,
    pipeline_extrapolate_velocity: ComputePipelineHandle,
    pipeline_advect_particles: ComputePipelineHandle,

    pipeline_density_projection_gather: ComputePipelineHandle,

    max_num_particles: u32,

    is_first_step: Cell<bool>,
}

static mut GROUP_LAYOUT_RENDERER: Option<BindGroupLayoutWithDesc> = None;

#[repr(C)]
#[derive(Clone, Copy)]
struct ParticlePositionLl {
    // Particle positions are in grid space to simplify shader computation
    // (no scaling/translation needed until we're rendering or interacting with other objects!)
    position: cgmath::Point3<f32>,
    linked_list_next: u32,
}
unsafe impl bytemuck::Pod for ParticlePositionLl {}
unsafe impl bytemuck::Zeroable for ParticlePositionLl {}

impl HybridFluid {
    // particles are distributed 2x2x2 within a single gridcell
    // (seems to be widely accepted as the default. Houdini seems to have this configurable from 4-16, maybe worth experimenting with it! (todo))
    pub const PARTICLES_PER_GRID_CELL: u32 = 8;

    pub fn new(
        device: &wgpu::Device,
        grid_dimension: wgpu::Extent3d,
        max_num_particles: u32,
        shader_dir: &ShaderDirectory,
        pipeline_manager: &mut PipelineManager,
        per_frame_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        // Resources
        let simulation_properties_uniformbuffer = UniformBuffer::new(device);
        let particles_position_llindex = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Buffer: Particles position & llindex"),
            size: max_num_particles as u64 * std::mem::size_of::<ParticlePositionLl>() as u64,
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
            mapped_at_creation: false,
        });
        let particles_velocity_x = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Buffer: Particles velocity X"),
            size: max_num_particles as u64 * std::mem::size_of::<cgmath::Vector4<f32>>() as u64,
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
            mapped_at_creation: false,
        });
        let particles_velocity_y = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Buffer: Particles velocity Y"),
            size: max_num_particles as u64 * std::mem::size_of::<cgmath::Vector4<f32>>() as u64,
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
            mapped_at_creation: false,
        });
        let particles_velocity_z = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Buffer: Particles velocity Z"),
            size: max_num_particles as u64 * std::mem::size_of::<cgmath::Vector4<f32>>() as u64,
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
            mapped_at_creation: false,
        });

        // TODO:
        // Various sources, old and new, claim that on Nvidia hardware 3D textures are actually 2d slices!
        // http://www-ppl.ist.osaka-u.ac.jp/research/papers/201405_sugimoto_pc.pdf
        // https://www.sciencedirect.com/science/article/pii/S2468502X1730027X#fig1
        // https://forum.unity.com/threads/improving-performance-of-3d-textures-using-texture-arrays.725384/#post-4849571
        // For Intel this is directly documented
        // https://www.x.org/docs/intel/BYT/intel_os_gfx_prm_vol5_-_memory_views.pdf
        // Wasn't able to find anything on AMD, but there is sources implying the layered nature
        // ("Two bilinear fetches are required when sampling from a volume texture with bilinear
        // filtering.")[http://amd-dev.wpengine.netdna-cdn.com/wordpress/media/2013/05/GCNPerformanceTweets.pdf]

        let create_volume_texture_desc = |label: &'static str, format: wgpu::TextureFormat| -> wgpu::TextureDescriptor {
            wgpu::TextureDescriptor {
                label: Some(label),
                size: grid_dimension,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format,
                usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::STORAGE,
            }
        };
        // TODO: Reuse volumes to safe memory, not all are used simultaneously.
        let volume_velocity_x = device.create_texture(&create_volume_texture_desc("Velocity Volume X", wgpu::TextureFormat::R32Float));
        let volume_velocity_y = device.create_texture(&create_volume_texture_desc("Velocity Volume Y", wgpu::TextureFormat::R32Float));
        let volume_velocity_z = device.create_texture(&create_volume_texture_desc("Velocity Volume Z", wgpu::TextureFormat::R32Float));
        let volume_linked_lists = device.create_texture(&create_volume_texture_desc("Linked Lists Volume", wgpu::TextureFormat::R32Uint));
        let volume_marker = device.create_texture(&create_volume_texture_desc("Marker Grid", wgpu::TextureFormat::R8Snorm));

        let volume_density = device.create_texture(&create_volume_texture_desc("Density Volume", wgpu::TextureFormat::R32Float));
        let volume_pressure = device.create_texture(&create_volume_texture_desc("Pressure Volume", wgpu::TextureFormat::R32Float)); // Previous frame is required for this one!
        let volume_pcg_residual = device.create_texture(&create_volume_texture_desc("PCG Solve Residual", wgpu::TextureFormat::R32Float));
        let volume_pcg_auxiliary = device.create_texture(&create_volume_texture_desc("PCG Solve Auxiliary", wgpu::TextureFormat::R32Float));
        let volume_pcg_auxiliary_temp = device.create_texture(&create_volume_texture_desc("PCG Solve Auxiliary Temp", wgpu::TextureFormat::R32Float));
        let volume_pcg_search = device.create_texture(&create_volume_texture_desc("Pressure Solve Search", wgpu::TextureFormat::R32Float));

        let num_cells = (grid_dimension.width * grid_dimension.height * grid_dimension.depth) as u64;
        let dotproduct_reduce_step_buffers = [
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Buffer: DotProduct Reduce 0"),
                size: num_cells * std::mem::size_of::<f32>() as u64,
                usage: wgpu::BufferUsage::STORAGE,
                mapped_at_creation: false,
            }),
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Buffer: DotProduct Reduce 1"),
                size: num_cells * std::mem::size_of::<f32>() as u64 / Self::DOTPRODUCT_REDUCE_REDUCTION_PER_STEP as u64,
                usage: wgpu::BufferUsage::STORAGE,
                mapped_at_creation: false,
            }),
        ];
        let dotproduct_reduce_result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Buffer: DotProduct Result"),
            size: 4 * std::mem::size_of::<f32>() as u64,
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::UNIFORM,
            mapped_at_creation: false,
        });

        // Resource views
        let volume_velocity_view_x = volume_velocity_x.create_view(&Default::default());
        let volume_velocity_view_y = volume_velocity_y.create_view(&Default::default());
        let volume_velocity_view_z = volume_velocity_z.create_view(&Default::default());
        let volume_linked_lists_view = volume_linked_lists.create_view(&Default::default());
        let volume_marker_view = volume_marker.create_view(&Default::default());

        let volume_pressure_view = volume_pressure.create_view(&Default::default());
        let volume_density_view = volume_density.create_view(&Default::default());
        let volume_pcg_residual_view = volume_pcg_residual.create_view(&Default::default());
        let volume_pcg_auxiliary_view = volume_pcg_auxiliary.create_view(&Default::default());
        let volume_pcg_auxiliary_temp_view = volume_pcg_auxiliary_temp.create_view(&Default::default());
        let volume_pcg_search_view = volume_pcg_search.create_view(&Default::default());

        // Layouts
        let group_layout_uniform = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::uniform())
            .create(device, "BindGroupLayout: HybridFluid Uniform");
        let group_layout_transfer_velocity = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::buffer(false)) // particles, position llindex
            .next_binding_compute(binding_glsl::buffer(true)) // particles, velocity component
            .next_binding_compute(binding_glsl::uimage3d(wgpu::TextureFormat::R32Uint, false)) // linkedlist_volume
            .next_binding_compute(binding_glsl::image3d(wgpu::TextureFormat::R8Snorm, false)) // marker volume
            .next_binding_compute(binding_glsl::image3d(wgpu::TextureFormat::R32Float, false)) // velocity component
            .create(device, "BindGroupLayout: Transfer velocity from Particles to Volume(s)");
        let group_layout_write_velocity = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::image3d(wgpu::TextureFormat::R32Float, false)) // velocityX
            .next_binding_compute(binding_glsl::image3d(wgpu::TextureFormat::R32Float, false)) // velocityY
            .next_binding_compute(binding_glsl::image3d(wgpu::TextureFormat::R32Float, false)) // velocityZ
            .next_binding_compute(binding_glsl::texture3D()) // marker volume
            .next_binding_compute(binding_glsl::texture2D()) // pressure
            .create(device, "BindGroupLayout: Write to Velocity");
        let group_layout_advect_particles = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::texture2D()) // velocityX
            .next_binding_compute(binding_glsl::texture2D()) // velocityY
            .next_binding_compute(binding_glsl::texture2D()) // velocityZ
            .next_binding_compute(binding_glsl::image3d(wgpu::TextureFormat::R8Snorm, false)) // marker volume
            .next_binding_compute(binding_glsl::uimage3d(wgpu::TextureFormat::R32Uint, false)) // linkedlist_volume
            .next_binding_compute(binding_glsl::buffer(false)) // particles, position llindex
            .next_binding_compute(binding_glsl::buffer(false)) // particles, velocityX
            .next_binding_compute(binding_glsl::buffer(false)) // particles, velocityY
            .next_binding_compute(binding_glsl::buffer(false)) // particles, velocityZ
            .create(device, "BindGroupLayout: Advect to Particles");

        let group_layout_read_macgrid = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::texture3D()) // velocityX
            .next_binding_compute(binding_glsl::texture3D()) // velocityY
            .next_binding_compute(binding_glsl::texture3D()) // velocityZ
            .next_binding_compute(binding_glsl::texture3D()) // marker volume
            .create(device, "BindGroupLayout: Read MAC Grid");
        let group_layout_pressure_solve_init = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::image3d(wgpu::TextureFormat::R32Float, false))
            .next_binding_compute(binding_glsl::image3d(wgpu::TextureFormat::R32Float, false))
            .create(device, "BindGroupLayout: Pressure solver init");
        let group_layout_apply_coeff = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::buffer(false))
            .next_binding_compute(binding_glsl::texture3D())
            .create(device, "BindGroupLayout: Apply coeff matrix & start dot product");
        let group_layout_pressure_solve_reduce = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::buffer(true)) // source
            .next_binding_compute(binding_glsl::buffer(false)) // dest
            .create(device, "BindGroupLayout: Pressure solver dot product reduce");
        let group_layout_pressure_preconditioner = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::buffer(false))
            .next_binding_compute(binding_glsl::texture3D())
            .next_binding_compute(binding_glsl::image3d(wgpu::TextureFormat::R32Float, false))
            .next_binding_compute(binding_glsl::texture3D())
            .create(device, "BindGroupLayout: Pressure solver preconditioner");
        let group_layout_pressure_update_volume = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::image3d(wgpu::TextureFormat::R32Float, false))
            .next_binding_compute(binding_glsl::texture3D())
            .next_binding_compute(binding_glsl::uniform())
            .create(device, "BindGroupLayout: Generic volume update");
        let group_layout_pressure_update_pressure_and_residual = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::image3d(wgpu::TextureFormat::R32Float, false))
            .next_binding_compute(binding_glsl::image3d(wgpu::TextureFormat::R32Float, false))
            .next_binding_compute(binding_glsl::texture3D())
            .next_binding_compute(binding_glsl::uniform())
            .create(device, "BindGroupLayout: PCG update pressure and residual");

        let group_layout_density_projection_gather = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::buffer(false)) // particles, position llindex
            .next_binding_compute(binding_glsl::utexture3D()) // linkedlist_volume
            .next_binding_compute(binding_glsl::image3d(wgpu::TextureFormat::R8Snorm, false)) // marker volume
            .next_binding_compute(binding_glsl::image3d(wgpu::TextureFormat::R32Float, false)) // density volume
            .create(device, "BindGroupLayout: Transfer velocity from Particles to Volume(s)");

        // Bind groups.
        let bind_group_uniform = BindGroupBuilder::new(&group_layout_uniform)
            .resource(simulation_properties_uniformbuffer.binding_resource())
            .create(device, "BindGroup: HybridFluid Uniform");

        let bind_group_transfer_velocity = [
            BindGroupBuilder::new(&group_layout_transfer_velocity)
                .buffer(particles_position_llindex.slice(..))
                .buffer(particles_velocity_x.slice(..))
                .texture(&volume_linked_lists_view)
                .texture(&volume_marker_view)
                .texture(&volume_velocity_view_x)
                .create(device, "BindGroup: Transfer velocity to volume X"),
            BindGroupBuilder::new(&group_layout_transfer_velocity)
                .buffer(particles_position_llindex.slice(..))
                .buffer(particles_velocity_y.slice(..))
                .texture(&volume_linked_lists_view)
                .texture(&volume_marker_view)
                .texture(&volume_velocity_view_y)
                .create(device, "BindGroup: Transfer velocity to volume Y"),
            BindGroupBuilder::new(&group_layout_transfer_velocity)
                .buffer(particles_position_llindex.slice(..))
                .buffer(particles_velocity_z.slice(..))
                .texture(&volume_linked_lists_view)
                .texture(&volume_marker_view)
                .texture(&volume_velocity_view_z)
                .create(device, "BindGroup: Transfer velocity to volume Z"),
        ];

        let bind_group_write_velocity = BindGroupBuilder::new(&group_layout_write_velocity)
            .texture(&volume_velocity_view_x)
            .texture(&volume_velocity_view_y)
            .texture(&volume_velocity_view_z)
            .texture(&volume_marker_view)
            .texture(&volume_pressure_view)
            .create(device, "BindGroup: Write to Velocity Grid");
        let bind_group_advect_particles = BindGroupBuilder::new(&group_layout_advect_particles)
            .texture(&volume_velocity_view_x)
            .texture(&volume_velocity_view_y)
            .texture(&volume_velocity_view_z)
            .texture(&volume_marker_view)
            .texture(&volume_linked_lists_view)
            .buffer(particles_position_llindex.slice(..))
            .buffer(particles_velocity_x.slice(..))
            .buffer(particles_velocity_y.slice(..))
            .buffer(particles_velocity_z.slice(..))
            .create(device, "BindGroup: Write to Particles");

        let bind_group_read_mac_grid = BindGroupBuilder::new(&group_layout_read_macgrid)
            .texture(&volume_velocity_view_x)
            .texture(&volume_velocity_view_y)
            .texture(&volume_velocity_view_z)
            .texture(&volume_marker_view)
            .create(device, "BindGroup: Read MAC Grid");
        let bind_group_pressure_compute_divergence = BindGroupBuilder::new(&group_layout_pressure_solve_init)
            .texture(&volume_pcg_residual_view)
            .texture(&volume_pressure_view)
            .create(device, "BindGroup: Compute initial residual");
        let bind_group_pressure_apply_coeff = BindGroupBuilder::new(&group_layout_apply_coeff)
            .buffer(dotproduct_reduce_step_buffers[0].slice(..))
            .texture(&volume_pcg_search_view)
            .create(device, "BindGroup: Apply coeff matrix & start dot product");
        let bind_group_pressure_preconditioner = [
            BindGroupBuilder::new(&group_layout_pressure_preconditioner)
                .buffer(dotproduct_reduce_step_buffers[0].slice(..))
                .texture(&volume_pcg_residual_view)
                .texture(&volume_pcg_auxiliary_temp_view)
                .texture(&volume_pcg_residual_view)
                .create(device, "BindGroup: Preconditioner, Step 1"),
            BindGroupBuilder::new(&group_layout_pressure_preconditioner)
                .buffer(dotproduct_reduce_step_buffers[0].slice(..))
                .texture(&volume_pcg_residual_view)
                .texture(&volume_pcg_auxiliary_view)
                .texture(&volume_pcg_auxiliary_temp_view)
                .create(device, "BindGroup: Preconditioner, Step 2"),
            BindGroupBuilder::new(&group_layout_pressure_preconditioner)
                .buffer(dotproduct_reduce_step_buffers[0].slice(..))
                .texture(&volume_pcg_residual_view)
                .texture(&volume_pcg_search_view)
                .texture(&volume_pcg_auxiliary_temp_view)
                .create(device, "BindGroup: Preconditioner, Step 2, to search"),
        ];
        let bind_group_pressure_dotproduct_reduce = [
            BindGroupBuilder::new(&group_layout_pressure_solve_reduce)
                .buffer(dotproduct_reduce_step_buffers[0].slice(..))
                .buffer(dotproduct_reduce_step_buffers[1].slice(..))
                .create(device, "BindGroup: Pressure Solve, Reduce 0"),
            BindGroupBuilder::new(&group_layout_pressure_solve_reduce)
                .buffer(dotproduct_reduce_step_buffers[1].slice(..))
                .buffer(dotproduct_reduce_step_buffers[0].slice(..))
                .create(device, "BindGroup: Pressure Solve, Reduce 1"),
        ];
        let bind_group_pressure_dotproduct_final = [
            BindGroupBuilder::new(&group_layout_pressure_solve_reduce)
                .buffer(dotproduct_reduce_step_buffers[0].slice(..))
                .buffer(dotproduct_reduce_result_buffer.slice(..))
                .create(device, "BindGroup: Pressure Solve, Reduce Final 0"),
            BindGroupBuilder::new(&group_layout_pressure_solve_reduce)
                .buffer(dotproduct_reduce_step_buffers[1].slice(..))
                .buffer(dotproduct_reduce_result_buffer.slice(..))
                .create(device, "BindGroup: Pressure Solve, Reduce Final 1"),
        ];

        let bind_group_pressure_update_pressure_and_residual = BindGroupBuilder::new(&group_layout_pressure_update_pressure_and_residual)
            .texture(&volume_pressure_view)
            .texture(&volume_pcg_residual_view)
            .texture(&volume_pcg_search_view)
            .buffer(dotproduct_reduce_result_buffer.slice(..))
            .create(device, "BindGroup: Pressure update pressure and residual");
        let bind_group_pressure_update_search = BindGroupBuilder::new(&group_layout_pressure_update_volume)
            .texture(&volume_pcg_search_view)
            .texture(&volume_pcg_auxiliary_view)
            .buffer(dotproduct_reduce_result_buffer.slice(..))
            .create(device, "BindGroup: Pressure update search");

        let bind_group_density_projection_gather = BindGroupBuilder::new(&group_layout_density_projection_gather)
            .buffer(particles_position_llindex.slice(..))
            .texture(&volume_linked_lists_view)
            .texture(&volume_marker_view)
            .texture(&volume_density_view)
            .create(device, "BindGroup: Density projection gather");

        let bind_group_renderer = BindGroupBuilder::new(&Self::get_or_create_group_layout_renderer(device))
            .buffer(particles_position_llindex.slice(..))
            .buffer(particles_velocity_x.slice(..))
            .buffer(particles_velocity_y.slice(..))
            .buffer(particles_velocity_z.slice(..))
            .texture(&volume_velocity_view_x)
            .texture(&volume_velocity_view_y)
            .texture(&volume_velocity_view_z)
            .texture(&volume_marker_view)
            .texture(&volume_pressure_view)
            .texture(&volume_density_view)
            .create(device, "BindGroup: Fluid Renderers");

        // pipeline layouts.
        // Use same push constant range for all pipelines to improve internal Vulkan pipeline compatibility.
        let push_constant_ranges = &[wgpu::PushConstantRange {
            stages: wgpu::ShaderStage::COMPUTE,
            range: 0..8,
        }];

        let layout_transfer_velocity = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("HybridFluid, Transfer Velocity Pipeline Layout"),
            bind_group_layouts: &[
                per_frame_bind_group_layout,
                &group_layout_uniform.layout,
                &group_layout_transfer_velocity.layout,
            ],
            push_constant_ranges,
        }));
        let layout_write_volume = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("HybridFluid, Write Volume Pipeline Layout"),
            bind_group_layouts: &[
                per_frame_bind_group_layout,
                &group_layout_uniform.layout,
                &group_layout_write_velocity.layout,
            ],
            push_constant_ranges,
        }));
        let layout_pressure_update_volume = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("HybridFluid, Update Volume Pipeline Layout"),
            bind_group_layouts: &[
                per_frame_bind_group_layout,
                &group_layout_read_macgrid.layout,
                &group_layout_pressure_update_volume.layout,
            ],
            push_constant_ranges,
        }));
        let layout_pressure_preconditioner = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("HybridFluid, Pressure Solve Precondition Pipeline Layout"),
            bind_group_layouts: &[
                per_frame_bind_group_layout,
                &group_layout_read_macgrid.layout,
                &group_layout_pressure_preconditioner.layout,
            ],
            push_constant_ranges,
        }));
        let layout_pressure_update_pressure_and_residual = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("HybridFluid, Pressure Solve Update P&R Pipeline Layout"),
            bind_group_layouts: &[
                per_frame_bind_group_layout,
                &group_layout_read_macgrid.layout,
                &group_layout_pressure_update_pressure_and_residual.layout,
            ],
            push_constant_ranges,
        }));
        let layout_pressure_solve_init = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("HybridFluid, Pressure Solve Init Pipeline Layout"),
            bind_group_layouts: &[
                per_frame_bind_group_layout,
                &group_layout_read_macgrid.layout,
                &group_layout_pressure_solve_init.layout,
            ],
            push_constant_ranges,
        }));

        let layout_pressure_apply_coeff = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("HybridFluid, Pressure Solve Apply Coeff Pipeline Layout"),
            bind_group_layouts: &[
                per_frame_bind_group_layout,
                &group_layout_read_macgrid.layout,
                &group_layout_apply_coeff.layout,
            ],
            push_constant_ranges,
        }));
        let layout_pressure_solve_reduce = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("HybridFluid, Pressure Solve Reduce Pipeline Layout"),
            bind_group_layouts: &[
                per_frame_bind_group_layout,
                &group_layout_read_macgrid.layout,
                &group_layout_pressure_solve_reduce.layout,
            ],
            push_constant_ranges,
        }));
        let layout_particles = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("HybridFluid, Particles Pipeline Layout"),
            bind_group_layouts: &[
                per_frame_bind_group_layout,
                &group_layout_uniform.layout,
                &group_layout_advect_particles.layout,
            ],
            push_constant_ranges,
        }));
        let layout_density_projection_gather = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("HybridFluid, Density Projection Gather Layout"),
            bind_group_layouts: &[
                per_frame_bind_group_layout,
                &group_layout_uniform.layout,
                &group_layout_density_projection_gather.layout,
            ],
            push_constant_ranges,
        }));

        HybridFluid {
            grid_dimension,

            particles_position_llindex,
            particles_velocity_x,
            particles_velocity_y,
            particles_velocity_z,
            simulation_properties_uniformbuffer,
            simulation_properties: SimulationPropertiesUniformBufferContent {
                num_particles: 0,
                gravity_grid: cgmath::vec3(0.0, -9.81, 0.0),
            },
            simulation_properties_dirty: Cell::new(true),

            bind_group_uniform,
            bind_group_transfer_velocity,
            bind_group_write_velocity,
            bind_group_advect_particles,

            bind_group_read_mac_grid,
            bind_group_pressure_compute_divergence,
            bind_group_pressure_preconditioner,

            bind_group_renderer,

            bind_group_pressure_apply_coeff,
            bind_group_pressure_dotproduct_reduce,
            bind_group_pressure_dotproduct_final,
            bind_group_pressure_update_pressure_and_residual,
            bind_group_pressure_update_search,
            bind_group_density_projection_gather,

            pipeline_transfer_clear: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(layout_transfer_velocity.clone(), Path::new("simulation/transfer_clear.comp")),
            ),
            pipeline_transfer_build_linkedlist: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(layout_transfer_velocity.clone(), Path::new("simulation/transfer_build_linkedlist.comp")),
            ),
            pipeline_transfer_gather_velocity: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(layout_transfer_velocity.clone(), Path::new("simulation/transfer_gather_velocity.comp")),
            ),

            pipeline_remove_divergence: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(layout_write_volume.clone(), Path::new("simulation/remove_divergence.comp")),
            ),
            pipeline_extrapolate_velocity: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(layout_write_volume.clone(), Path::new("simulation/extrapolate_velocity.comp")),
            ),
            pipeline_advect_particles: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(layout_particles.clone(), Path::new("simulation/advect_particles.comp")),
            ),

            pipeline_pressure_compute_divergence: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(
                    layout_pressure_solve_init.clone(),
                    Path::new("simulation/pressure_compute_divergence.comp"),
                ),
            ),
            pipeline_pressure_apply_preconditioner: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(
                    layout_pressure_preconditioner.clone(),
                    Path::new("simulation/pressure_apply_preconditioner.comp"),
                ),
            ),
            pipeline_pressure_reduce: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(layout_pressure_solve_reduce.clone(), Path::new("simulation/pressure_reduce.comp")),
            ),
            pipeline_pressure_apply_coeff: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(layout_pressure_apply_coeff.clone(), Path::new("simulation/pressure_apply_coeff.comp")),
            ),
            pipeline_pressure_update_pressure_and_residual: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(
                    layout_pressure_update_pressure_and_residual.clone(),
                    Path::new("simulation/pressure_update_pressure_and_residual.comp"),
                ),
            ),
            pipeline_pressure_update_search: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(layout_pressure_update_volume.clone(), Path::new("simulation/pressure_update_search.comp")),
            ),

            pipeline_density_projection_gather: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(
                    layout_density_projection_gather.clone(),
                    Path::new("simulation/density_projection_gather.comp"),
                ),
            ),

            max_num_particles,
            is_first_step: Cell::new(true),
        }
    }

    fn clamp_to_grid(&self, grid_cor: cgmath::Point3<f32>) -> cgmath::Point3<u32> {
        // Due to the design of the grid, the 0-1 range is reserved by solid cells and can't be filled.
        cgmath::Point3::new(
            self.grid_dimension.width.min(grid_cor.x as u32).max(1),
            self.grid_dimension.height.min(grid_cor.y as u32).max(1),
            self.grid_dimension.depth.min(grid_cor.z as u32).max(1),
        )
    }

    // Adds a cube of fluid. Coordinates are in grid space! Very slow operation!
    pub fn add_fluid_cube(&mut self, queue: &wgpu::Queue, min_grid: cgmath::Point3<f32>, max_grid: cgmath::Point3<f32>) {
        // align to whole cells for simplicity.
        let min_grid = self.clamp_to_grid(min_grid);
        let max_grid = self.clamp_to_grid(max_grid);
        let extent_cell = max_grid - min_grid;

        let mut num_new_particles = (extent_cell.x * extent_cell.y * extent_cell.z * Self::PARTICLES_PER_GRID_CELL) as u32;
        if self.max_num_particles < num_new_particles + self.simulation_properties.num_particles {
            error!(
                "Can't add {} particles, max is {}, current is {}",
                num_new_particles, self.max_num_particles, self.simulation_properties.num_particles
            );
            num_new_particles = self.max_num_particles - self.simulation_properties.num_particles;
        }
        info!("Adding {} new particles", num_new_particles);

        // Fill buffer with particle data
        let mut rng: rand::rngs::SmallRng = rand::SeedableRng::seed_from_u64((self.simulation_properties.num_particles + num_new_particles) as u64);
        let mut new_particles = Vec::new();
        new_particles.resize(
            num_new_particles as usize,
            ParticlePositionLl {
                position: cgmath::point3(0.0, 0.0, 0.0),
                linked_list_next: 0xFFFFFFFF,
            },
        );
        for (i, particle) in new_particles.iter_mut().enumerate() {
            let cell = cgmath::point3(
                (min_grid.x + i as u32 / Self::PARTICLES_PER_GRID_CELL % extent_cell.x) as f32,
                (min_grid.y + i as u32 / Self::PARTICLES_PER_GRID_CELL / extent_cell.x % extent_cell.y) as f32,
                (min_grid.z + i as u32 / Self::PARTICLES_PER_GRID_CELL / extent_cell.x / extent_cell.y) as f32,
            );

            let sample_idx = i as u32 % Self::PARTICLES_PER_GRID_CELL;

            // pure random
            // let offset = rng.gen::<cgmath::Vector3<f32>>();
            // pure regular
            // let offset = cgmath::vec3(
            //     (sample_idx % 2) as f32 + 0.5,
            //     (sample_idx / 2 % 2) as f32 + 0.5,
            //     (sample_idx / 4 % 2) as f32 + 0.5,
            // ) * 0.5;
            // stratified
            let offset = cgmath::vec3((sample_idx % 2) as f32, (sample_idx / 2 % 2) as f32, (sample_idx / 4 % 2) as f32) * 0.5
                + rng.gen::<cgmath::Vector3<f32>>() * 0.5;

            particle.position = cell + offset;
        }

        let particle_size = std::mem::size_of::<ParticlePositionLl>() as u64;
        queue.write_buffer(
            &self.particles_position_llindex,
            self.simulation_properties.num_particles as u64 * particle_size,
            bytemuck::cast_slice(&new_particles),
        );

        // Clear velocities:
        // wgpu-rs doesn't zero initialize yet (bug/missing feature impl)
        // https://github.com/gfx-rs/wgpu/issues/563
        let offset_velocity_buffer = self.simulation_properties.num_particles as u64 * std::mem::size_of::<cgmath::Vector4<f32>>() as u64;
        let zero_velocity = vec![0 as u8; num_new_particles as usize * std::mem::size_of::<cgmath::Vector4<f32>>()];
        queue.write_buffer(&self.particles_velocity_x, offset_velocity_buffer, &zero_velocity);
        queue.write_buffer(&self.particles_velocity_y, offset_velocity_buffer, &zero_velocity);
        queue.write_buffer(&self.particles_velocity_z, offset_velocity_buffer, &zero_velocity);

        self.simulation_properties.num_particles += num_new_particles;
        self.simulation_properties_dirty.set(true);
    }

    pub fn set_gravity_grid(&mut self, gravity: cgmath::Vector3<f32>) {
        self.simulation_properties.gravity_grid = gravity;
        self.simulation_properties_dirty.set(true);
    }

    pub fn num_particles(&self) -> u32 {
        self.simulation_properties.num_particles
    }

    pub fn get_or_create_group_layout_renderer(device: &wgpu::Device) -> &BindGroupLayoutWithDesc {
        unsafe {
            GROUP_LAYOUT_RENDERER.get_or_insert_with(|| {
                BindGroupLayoutBuilder::new()
                    .next_binding_vertex(binding_glsl::buffer(true)) // particles, position llindex
                    .next_binding_vertex(binding_glsl::buffer(true)) // particles, velocityX
                    .next_binding_vertex(binding_glsl::buffer(true)) // particles, velocityY
                    .next_binding_vertex(binding_glsl::buffer(true)) // particles, velocityZ
                    .next_binding_vertex(binding_glsl::texture3D()) // velocityX
                    .next_binding_vertex(binding_glsl::texture3D()) // velocityY
                    .next_binding_vertex(binding_glsl::texture3D()) // velocityZ
                    .next_binding_vertex(binding_glsl::texture3D()) // marker
                    .next_binding_vertex(binding_glsl::texture3D()) // pressure
                    .next_binding_vertex(binding_glsl::texture3D()) // density
                    .create(device, "BindGroupLayout: ParticleRenderer")
            })
        }
    }

    pub fn bind_group_renderer(&self) -> &wgpu::BindGroup {
        &self.bind_group_renderer
    }

    pub fn grid_dimension(&self) -> wgpu::Extent3d {
        self.grid_dimension
    }

    const COMPUTE_LOCAL_SIZE_FLUID: wgpu::Extent3d = wgpu::Extent3d {
        width: 8,
        height: 8,
        depth: 8,
    };
    const COMPUTE_LOCAL_SIZE_PARTICLES: u32 = 512;
    const DOTPRODUCT_RESULTMODE_REDUCE: u32 = 0;
    const DOTPRODUCT_RESULTMODE_INIT: u32 = 1;
    const DOTPRODUCT_RESULTMODE_ALPHA: u32 = 2;
    const DOTPRODUCT_RESULTMODE_BETA: u32 = 3;

    const COMPUTE_LOCAL_SIZE_DOTPRODUCT_REDUCE: u32 = 1024;
    const DOTPRODUCT_READS_PER_STEP: u32 = 16; // 32 was distinctively slower, 16 about same as than 8, 4 clearly slower (gtx1070 ti)
    const DOTPRODUCT_REDUCE_REDUCTION_PER_STEP: u32 = Self::COMPUTE_LOCAL_SIZE_DOTPRODUCT_REDUCE * Self::DOTPRODUCT_READS_PER_STEP;

    fn reduce_add<'a, 'b: 'a>(&'b self, cpass: &mut wgpu::ComputePass<'a>, pipeline_manager: &'a PipelineManager, result_mode: u32) {
        // reduce.
        let mut source_buffer_index = 0;
        cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_pressure_reduce));
        let mut num_entries_remaining = (self.grid_dimension.width * self.grid_dimension.height * self.grid_dimension.depth) as u32;
        while num_entries_remaining > Self::DOTPRODUCT_REDUCE_REDUCTION_PER_STEP {
            cpass.set_bind_group(2, &self.bind_group_pressure_dotproduct_reduce[source_buffer_index], &[]);
            cpass.set_push_constants(0, &[Self::DOTPRODUCT_RESULTMODE_REDUCE, num_entries_remaining]);
            cpass.dispatch(
                wgpu_utils::compute_group_size_1d(
                    num_entries_remaining / Self::DOTPRODUCT_READS_PER_STEP,
                    Self::COMPUTE_LOCAL_SIZE_DOTPRODUCT_REDUCE,
                ),
                1,
                1,
            );
            source_buffer_index = 1 - source_buffer_index;
            num_entries_remaining /= Self::DOTPRODUCT_REDUCE_REDUCTION_PER_STEP;
        }
        // final
        cpass.set_bind_group(2, &self.bind_group_pressure_dotproduct_final[source_buffer_index], &[]);
        cpass.set_push_constants(0, &[result_mode, num_entries_remaining]);
        cpass.dispatch(
            wgpu_utils::compute_group_size_1d(num_entries_remaining, Self::COMPUTE_LOCAL_SIZE_DOTPRODUCT_REDUCE),
            1,
            1,
        );
    }

    pub fn step(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline_manager: &PipelineManager,
        queue: &wgpu::Queue,
        per_frame_bind_group: &wgpu::BindGroup,
    ) {
        if self.simulation_properties_dirty.get() {
            self.simulation_properties_uniformbuffer.update_content(queue, self.simulation_properties);
            self.simulation_properties_dirty.set(false);
        }

        let grid_work_groups = wgpu_utils::compute_group_size(self.grid_dimension, Self::COMPUTE_LOCAL_SIZE_FLUID);
        let particle_work_groups = wgpu_utils::compute_group_size_1d(self.simulation_properties.num_particles, Self::COMPUTE_LOCAL_SIZE_PARTICLES);

        let mut cpass = encoder.begin_compute_pass();
        cpass.set_bind_group(0, &per_frame_bind_group, &[]);
        cpass.set_bind_group(1, &self.bind_group_uniform, &[]);

        // mostly grouped by layouts.
        {
            for i in 0..3 {
                cpass.set_bind_group(2, &self.bind_group_transfer_velocity[i], &[]);

                // clear linkedlist grid & marker (only on first run)
                cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_transfer_clear));
                cpass.set_push_constants(0, &[i as u32]);
                cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);

                // Create particle linked lists by writing heads in dual grids
                cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_transfer_build_linkedlist));
                cpass.dispatch(particle_work_groups, 1, 1);

                // Gather velocities in velocity grid and apply global forces.
                cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_transfer_gather_velocity));
                cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);
            }
        }
        // Pressure solve with PCG
        // For optimization various steps are collapsed as far as possible to avoid expensive buffer/texture read/writes
        // This makes the algorithm a lot faster but also a bit harder to read.
        {
            cpass.set_bind_group(1, &self.bind_group_read_mac_grid, &[]);

            // Compute divergence (b) and the initial residual field (r)
            // We use pressure from last frame, but set explicitly set all pressure values to zero wherever there is not fluid right now.
            // This is done in order to prevent having results from many frames ago influence results for upcoming frames.
            // In first step overall we instruct to use a fresh pressure buffer.
            const DIVERGENCE_FIRST_STEP: u32 = 0;
            const DIVERGENCE_NOT_FIRST_STEP: u32 = 1;
            cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_pressure_compute_divergence));
            // Clear pressures on first step.
            // wgpu-rs doesn't zero initialize yet (bug/missing feature impl)
            // Most resources are derived from particles which we initialize ourselves, but not pressure where we use the previous step to kickstart the solver
            // https://github.com/gfx-rs/wgpu/issues/563
            if self.is_first_step.get() {
                cpass.set_push_constants(0, &[DIVERGENCE_FIRST_STEP]);
            } else {
                cpass.set_push_constants(0, &[DIVERGENCE_NOT_FIRST_STEP]);
            }
            cpass.set_bind_group(2, &self.bind_group_pressure_compute_divergence, &[]);
            cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);

            // Apply preconditioner on (r), store result to search vector (s) and start dotproduct of <s; r>
            // Note that we don't use the auxillary vector here as in-between storage!
            const PRECONDITIONER_PASS0: u32 = 0;
            const PRECONDITIONER_PASS1: u32 = 1;
            const PRECONDITIONER_PASS1_SET_UNUSED_TO_ZERO: u32 = 3;
            cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_pressure_apply_preconditioner));
            cpass.set_push_constants(0, &[0]);
            cpass.set_push_constants(0, &[PRECONDITIONER_PASS0]);
            cpass.set_bind_group(2, &self.bind_group_pressure_preconditioner[0], &[]);
            cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);
            cpass.set_push_constants(0, &[PRECONDITIONER_PASS1_SET_UNUSED_TO_ZERO]);
            cpass.set_bind_group(2, &self.bind_group_pressure_preconditioner[2], &[]);
            cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);
            // Init sigma to dotproduct of search vector (s) and residual (r)
            self.reduce_add(&mut cpass, pipeline_manager, Self::DOTPRODUCT_RESULTMODE_INIT);

            // Solver iterations ...
            const NUM_ITERATIONS: u32 = 16;
            let mut i = 0;
            loop {
                // Apply cell relationships to search vector (i.e. multiply s with A)
                // The dot product is applied to the result (denoted as z in Bridson's book) and the search vector (s), i.e. compute <s; As>
                cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_pressure_apply_coeff));
                cpass.set_bind_group(2, &self.bind_group_pressure_apply_coeff, &[]);
                cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);
                // finish dotproduct of auxiliary field (z) and search field (s)
                self.reduce_add(&mut cpass, pipeline_manager, Self::DOTPRODUCT_RESULTMODE_ALPHA);

                // update pressure field (p) and residual field (r)
                const PRUPDATE_LAST_ITERATION: u32 = 1;
                cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_pressure_update_pressure_and_residual));
                if i == NUM_ITERATIONS {
                    cpass.set_push_constants(0, &[PRUPDATE_LAST_ITERATION]);
                } else {
                    cpass.set_push_constants(0, &[0]);
                }
                cpass.set_bind_group(2, &self.bind_group_pressure_update_pressure_and_residual, &[]);
                cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);
                if i == NUM_ITERATIONS {
                    break;
                }

                // Apply preconditioner on (r), store result to auxillary (z) and start dotproduct of <z; r>
                cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_pressure_apply_preconditioner));
                cpass.set_push_constants(0, &[PRECONDITIONER_PASS0]);
                cpass.set_bind_group(2, &self.bind_group_pressure_preconditioner[0], &[]);
                cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);
                cpass.set_push_constants(0, &[PRECONDITIONER_PASS1]);
                cpass.set_bind_group(2, &self.bind_group_pressure_preconditioner[1], &[]);
                cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);
                // finish dotproduct of auxiliary field (z) and residual field (r)
                self.reduce_add(&mut cpass, pipeline_manager, Self::DOTPRODUCT_RESULTMODE_BETA);

                // Update search vector
                cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_pressure_update_search));
                cpass.set_bind_group(2, &self.bind_group_pressure_update_search, &[]);
                cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);

                i += 1;
            }
        }

        cpass.set_bind_group(1, &self.bind_group_uniform, &[]);
        {
            cpass.set_bind_group(2, &self.bind_group_write_velocity, &[]);

            // Make velocity grid divergence free
            cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_remove_divergence));
            cpass.set_bind_group(2, &self.bind_group_write_velocity, &[]);
            cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);

            // Extrapolate velocity
            // can only do a single extrapolation since we can't change cell types without double buffering
            // (this makes the extrapolation a bit heavier since it needs to sample all 8 diagonals as well)
            cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_extrapolate_velocity));
            cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);
        }
        {
            // Clear Marker & linked list.
            cpass.set_bind_group(2, &self.bind_group_transfer_velocity[0], &[]);
            cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_transfer_clear));
            cpass.set_push_constants(0, &[0]);
            cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);
        }
        {
            // Advect particles with grid and write new linked list for density gather.
            cpass.set_bind_group(2, &self.bind_group_advect_particles, &[]);
            cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_advect_particles));
            cpass.dispatch(particle_work_groups, 1, 1);
        }
        {
            // Compute density grid by another gather pass
            cpass.set_bind_group(2, &self.bind_group_density_projection_gather, &[]);
            cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_density_projection_gather));
            cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);
        }
        self.is_first_step.set(false);
    }
}
