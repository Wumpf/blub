use super::pressure_solver::*;
use crate::{
    scene::voxelization::SceneVoxelization,
    wgpu_utils::{self, binding_builder::*, binding_glsl, pipelines::*, shader::*, uniformbuffer::*},
};
use rand::prelude::*;
use std::{collections::VecDeque, path::Path, rc::Rc, time::Duration};
use wgpu_profiler::{wgpu_profiler, GpuProfiler};

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

    pressure_solver: PressureSolver,
    pressure_field_from_velocity: PressureField,
    pressure_field_from_density: PressureField,

    volume_linked_lists: wgpu::Texture,
    volume_marker: wgpu::Texture,
    volume_debug: Option<wgpu::Texture>,

    particles_position_llindex: wgpu::Buffer,
    particles_position_llindex_tmp: wgpu::Buffer,
    particle_binning_atomic_counter: wgpu::Buffer,
    simulation_properties_uniformbuffer: UniformBuffer<SimulationPropertiesUniformBufferContent>,
    simulation_properties: SimulationPropertiesUniformBufferContent,

    bind_group_general: wgpu::BindGroup,
    bind_group_transfer_velocity: [wgpu::BindGroup; 3],
    bind_group_divergence_compute: wgpu::BindGroup,
    bind_group_divergence_projection_write_velocity: wgpu::BindGroup,
    bind_group_advect_particles: wgpu::BindGroup,
    bind_group_binning: wgpu::BindGroup,
    bind_group_density_projection_gather_error: wgpu::BindGroup,
    bind_group_density_projection_correct_particles: wgpu::BindGroup,
    bind_group_density_projection_write_velocity: wgpu::BindGroup,

    // The interface to any renderer of the fluid. Readonly access to relevant resources
    bind_group_renderer: wgpu::BindGroup,

    pipeline_transfer_clear: ComputePipelineHandle,
    pipeline_transfer_build_linkedlist: ComputePipelineHandle,
    pipeline_transfer_set_boundary_marker: ComputePipelineHandle,
    pipeline_transfer_gather_velocity: ComputePipelineHandle,
    pipeline_divergence_compute: ComputePipelineHandle,
    pipeline_divergence_remove: ComputePipelineHandle,
    pipeline_extrapolate_velocity: ComputePipelineHandle,
    pipeline_advect_particles: ComputePipelineHandle,
    pipeline_binning_count: ComputePipelineHandle,
    pipeline_binning_scan: ComputePipelineHandle,
    pipeline_binning_rewrite_particles: ComputePipelineHandle,
    pipeline_density_projection_gather_error: ComputePipelineHandle,
    pipeline_density_projection_position_change: ComputePipelineHandle,
    pipeline_density_projection_correct_particles: ComputePipelineHandle,

    max_num_particles: u32,
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
    // (seems to be widely accepted as the default. Houdini seems to have this configurable from 4-16, maybe worth experimenting with it! Note however, that the density error computation assumes this constant as well!)
    pub const PARTICLES_PER_GRID_CELL: u32 = 8;

    pub fn new(
        device: &wgpu::Device,
        grid_dimension: wgpu::Extent3d,
        max_num_particles: u32,
        shader_dir: &ShaderDirectory,
        pipeline_manager: &mut PipelineManager,
        global_bind_group_layout: &wgpu::BindGroupLayout,
        voxelization: &SceneVoxelization,
    ) -> Self {
        // Resources
        let simulation_properties_uniformbuffer = UniformBuffer::new(device);

        let create_particle_buffer = |label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: max_num_particles as u64 * std::mem::size_of::<ParticlePositionLl>() as u64,
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
                mapped_at_creation: false,
            })
        };

        let particles_position_llindex = create_particle_buffer("Buffer: Particles position & llindex");
        let particles_position_llindex_tmp = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Buffer: Particles position & llindex tmp"),
            size: max_num_particles as u64 * std::mem::size_of::<ParticlePositionLl>() as u64,
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
            mapped_at_creation: false,
        });
        let particles_velocity_x = create_particle_buffer("Buffer: Particles velocity X");
        let particles_velocity_y = create_particle_buffer("Buffer: Particles velocity Y");
        let particles_velocity_z = create_particle_buffer("Buffer: Particles velocity Z");
        let particle_binning_atomic_counter = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Buffer: Atomic counter for particle binning"),
            size: wgpu::BIND_BUFFER_ALIGNMENT,
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
            mapped_at_creation: false,
        });

        let create_volume_texture_desc = |label: &'static str, format: wgpu::TextureFormat| -> wgpu::TextureDescriptor {
            wgpu::TextureDescriptor {
                label: Some(label),
                size: grid_dimension,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format,
                usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::STORAGE | wgpu::TextureUsage::COPY_DST,
            }
        };
        // TODO: Reuse volumes to safe memory, not all are used simultaneously.
        let volume_velocity_x = device.create_texture(&create_volume_texture_desc("Velocity Volume X", wgpu::TextureFormat::R32Float));
        let volume_velocity_y = device.create_texture(&create_volume_texture_desc("Velocity Volume Y", wgpu::TextureFormat::R32Float));
        let volume_velocity_z = device.create_texture(&create_volume_texture_desc("Velocity Volume Z", wgpu::TextureFormat::R32Float));
        let volume_linked_lists = device.create_texture(&create_volume_texture_desc(
            "Linked Lists / Particle Binning Volume",
            wgpu::TextureFormat::R32Uint,
        ));
        let volume_marker = device.create_texture(&create_volume_texture_desc("Marker Grid", wgpu::TextureFormat::R8Snorm));
        let volume_debug = if cfg!(debug_assertions) {
            Some(device.create_texture(&create_volume_texture_desc("Debug Volume", wgpu::TextureFormat::R32Float)))
        } else {
            None
        };

        // Resource views
        let volume_velocity_view_x = volume_velocity_x.create_view(&Default::default());
        let volume_velocity_view_y = volume_velocity_y.create_view(&Default::default());
        let volume_velocity_view_z = volume_velocity_z.create_view(&Default::default());
        let volume_linked_lists_view = volume_linked_lists.create_view(&Default::default());
        let volume_marker_view = volume_marker.create_view(&Default::default());
        let volume_debug_view = match volume_debug {
            Some(ref volume) => Some(volume.create_view(&Default::default())),
            None => None,
        };

        // Layouts
        let group_layout_general = {
            let base_desc = BindGroupLayoutBuilder::new()
                .next_binding_compute(binding_glsl::uniform())
                .next_binding_compute(binding_glsl::texture3D());
            if volume_debug_view.is_some() {
                base_desc.next_binding_compute(binding_glsl::image3D(
                    wgpu::TextureFormat::R32Float,
                    wgpu::StorageTextureAccess::ReadWrite,
                ))
            } else {
                base_desc
            }
            .create(device, "BindGroupLayout: HybridFluid Uniform")
        };
        let group_layout_transfer_velocity = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::buffer(false)) // particles, position llindex
            .next_binding_compute(binding_glsl::buffer(true)) // particles, velocity component
            .next_binding_compute(binding_glsl::image3D(wgpu::TextureFormat::R32Uint, wgpu::StorageTextureAccess::ReadWrite)) // linkedlist_volume
            .next_binding_compute(binding_glsl::image3D(wgpu::TextureFormat::R8Snorm, wgpu::StorageTextureAccess::ReadWrite)) // marker volume
            .next_binding_compute(binding_glsl::image3D(
                wgpu::TextureFormat::R32Float,
                wgpu::StorageTextureAccess::ReadWrite,
            )) // velocity component
            .create(device, "BindGroupLayout: Transfer velocity from Particles to Volume(s)");
        let group_layout_divergence_compute = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::texture3D()) // marker volume
            .next_binding_compute(binding_glsl::texture3D()) // velocityX
            .next_binding_compute(binding_glsl::texture3D()) // velocityY
            .next_binding_compute(binding_glsl::texture3D()) // velocityZ
            .next_binding_compute(binding_glsl::image3D(
                wgpu::TextureFormat::R32Float,
                wgpu::StorageTextureAccess::ReadWrite,
            )) // divergence / initial residual
            .create(device, "BindGroupLayout: Compute Divergence");
        let group_layout_write_velocity_volume = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::texture3D()) // marker volume
            .next_binding_compute(binding_glsl::image3D(
                wgpu::TextureFormat::R32Float,
                wgpu::StorageTextureAccess::ReadWrite,
            )) // velocityX
            .next_binding_compute(binding_glsl::image3D(
                wgpu::TextureFormat::R32Float,
                wgpu::StorageTextureAccess::ReadWrite,
            )) // velocityY
            .next_binding_compute(binding_glsl::image3D(
                wgpu::TextureFormat::R32Float,
                wgpu::StorageTextureAccess::ReadWrite,
            )) // velocityZ
            .next_binding_compute(binding_glsl::texture3D()) // pressure
            .create(device, "BindGroupLayout: Write to Velocity");
        let group_layout_advect_particles = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::texture3D()) // velocityX
            .next_binding_compute(binding_glsl::texture3D()) // velocityY
            .next_binding_compute(binding_glsl::texture3D()) // velocityZ
            .next_binding_compute(binding_glsl::image3D(wgpu::TextureFormat::R8Snorm, wgpu::StorageTextureAccess::ReadWrite)) // marker volume
            .next_binding_compute(binding_glsl::image3D(wgpu::TextureFormat::R32Uint, wgpu::StorageTextureAccess::ReadWrite)) // linkedlist_volume
            .next_binding_compute(binding_glsl::buffer(false)) // particles, position llindex
            .next_binding_compute(binding_glsl::buffer(false)) // particles, velocityX
            .next_binding_compute(binding_glsl::buffer(false)) // particles, velocityY
            .next_binding_compute(binding_glsl::buffer(false)) // particles, velocityZ
            .create(device, "BindGroupLayout: Advect to Particles");

        let group_layout_binning = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::buffer(true)) // particles, position llindex
            .next_binding_compute(binding_glsl::buffer(true)) // particles, position llindex
            .next_binding_compute(binding_glsl::image3D(wgpu::TextureFormat::R32Uint, wgpu::StorageTextureAccess::ReadWrite)) // volume_particle_binning
            .next_binding_compute(binding_glsl::buffer(false)) // ParticleBinningAtomicCounter
            .create(device, "BindGroupLayout: Binning");
        let group_layout_density_projection_gather_error = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::buffer(false)) // particles, position llindex
            .next_binding_compute(binding_glsl::utexture3D()) // linkedlist_volume
            .next_binding_compute(binding_glsl::image3D(wgpu::TextureFormat::R8Snorm, wgpu::StorageTextureAccess::ReadWrite)) // marker volume
            .next_binding_compute(binding_glsl::image3D(
                wgpu::TextureFormat::R32Float,
                wgpu::StorageTextureAccess::ReadWrite,
            )) // density volume
            .create(device, "BindGroupLayout: Compute density error");
        let group_layout_density_projection_correct_particles = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::buffer(false)) // particles, position llindex
            .next_binding_compute(binding_glsl::texture3D()) // marker volume
            .next_binding_compute(binding_glsl::texture3D()) // velocityX
            .next_binding_compute(binding_glsl::texture3D()) // velocityY
            .next_binding_compute(binding_glsl::texture3D()) // velocityZ
            .create(device, "BindGroupLayout: Correct density error");

        let solver_config = SolverConfig {
            error_tolerance: 0.1,
            error_check_frequency: 4,
            max_num_iterations: 32,
        };
        let pressure_solver = PressureSolver::new(device, grid_dimension, shader_dir, pipeline_manager, &volume_marker_view);
        let pressure_field_from_velocity = PressureField::new("from velocity", device, grid_dimension, &pressure_solver, solver_config);
        let pressure_field_from_density = PressureField::new("from density", device, grid_dimension, &pressure_solver, solver_config);

        // Bind groups.
        let bind_group_general = {
            let base_desc = BindGroupBuilder::new(&group_layout_general)
                .resource(simulation_properties_uniformbuffer.binding_resource())
                .texture(voxelization.texture_view());
            match volume_debug_view.as_ref() {
                Some(volume_debug_view) => base_desc.texture(volume_debug_view),
                None => base_desc,
            }
            .create(device, "BindGroup: HybridFluid Uniform")
        };

        let bind_group_transfer_velocity = [
            BindGroupBuilder::new(&group_layout_transfer_velocity)
                .resource(particles_position_llindex.as_entire_binding())
                .resource(particles_velocity_x.as_entire_binding())
                .texture(&volume_linked_lists_view)
                .texture(&volume_marker_view)
                .texture(&volume_velocity_view_x)
                .create(device, "BindGroup: Transfer velocity to volume X, p-buffer"),
            BindGroupBuilder::new(&group_layout_transfer_velocity)
                .resource(particles_position_llindex.as_entire_binding())
                .resource(particles_velocity_y.as_entire_binding())
                .texture(&volume_linked_lists_view)
                .texture(&volume_marker_view)
                .texture(&volume_velocity_view_y)
                .create(device, "BindGroup: Transfer velocity to volume Y, p-buffer"),
            BindGroupBuilder::new(&group_layout_transfer_velocity)
                .resource(particles_position_llindex.as_entire_binding())
                .resource(particles_velocity_z.as_entire_binding())
                .texture(&volume_linked_lists_view)
                .texture(&volume_marker_view)
                .texture(&volume_velocity_view_z)
                .create(device, "BindGroup: Transfer velocity to volume Z, p-buffer"),
        ];
        let bind_group_divergence_compute = BindGroupBuilder::new(&group_layout_divergence_compute)
            .texture(&volume_marker_view)
            .texture(&volume_velocity_view_x)
            .texture(&volume_velocity_view_y)
            .texture(&volume_velocity_view_z)
            .texture(pressure_solver.residual_view())
            .create(device, "BindGroup: Compute divergence");
        let bind_group_divergence_projection_write_velocity = BindGroupBuilder::new(&group_layout_write_velocity_volume)
            .texture(&volume_marker_view)
            .texture(&volume_velocity_view_x)
            .texture(&volume_velocity_view_y)
            .texture(&volume_velocity_view_z)
            .texture(pressure_field_from_velocity.pressure_view())
            .create(device, "BindGroup: Write to Velocity Grid - divergence projection");
        let bind_group_density_projection_write_velocity = BindGroupBuilder::new(&group_layout_write_velocity_volume)
            .texture(&volume_marker_view)
            .texture(&volume_velocity_view_x)
            .texture(&volume_velocity_view_y)
            .texture(&volume_velocity_view_z)
            .texture(pressure_field_from_density.pressure_view())
            .create(device, "BindGroup: Write to Velocity Grid - density projection");
        // todo: Very ugly duplication
        let bind_group_advect_particles = BindGroupBuilder::new(&group_layout_advect_particles)
            .texture(&volume_velocity_view_x)
            .texture(&volume_velocity_view_y)
            .texture(&volume_velocity_view_z)
            .texture(&volume_marker_view)
            .texture(&volume_linked_lists_view)
            .resource(particles_position_llindex.as_entire_binding())
            .resource(particles_velocity_x.as_entire_binding())
            .resource(particles_velocity_y.as_entire_binding())
            .resource(particles_velocity_z.as_entire_binding())
            .create(device, "BindGroup: Write to Particles");

        let bind_group_binning = BindGroupBuilder::new(&group_layout_binning)
            .resource(particles_position_llindex.as_entire_binding())
            .resource(particles_position_llindex_tmp.as_entire_binding())
            .texture(&volume_linked_lists_view) // reused for binning counters
            .resource(particle_binning_atomic_counter.as_entire_binding())
            .create(device, "BindGroup: Binning");

        let bind_group_density_projection_gather_error = BindGroupBuilder::new(&group_layout_density_projection_gather_error)
            .resource(particles_position_llindex.as_entire_binding())
            .texture(&volume_linked_lists_view)
            .texture(&volume_marker_view)
            .texture(&pressure_solver.residual_view())
            .create(device, "BindGroup: Density projection gather 0");
        let bind_group_density_projection_correct_particles = BindGroupBuilder::new(&group_layout_density_projection_correct_particles)
            .resource(particles_position_llindex.as_entire_binding())
            .texture(&volume_marker_view)
            .texture(&volume_velocity_view_x)
            .texture(&volume_velocity_view_y)
            .texture(&volume_velocity_view_z)
            .create(device, "BindGroup: Density projection correct particles 0");
        let bind_group_renderer = {
            let bind_group_renderer_builder = BindGroupBuilder::new(&Self::get_or_create_group_layout_renderer(device))
                .resource(particles_position_llindex.as_entire_binding())
                .resource(particles_velocity_x.as_entire_binding())
                .resource(particles_velocity_y.as_entire_binding())
                .resource(particles_velocity_z.as_entire_binding())
                .texture(&volume_velocity_view_x)
                .texture(&volume_velocity_view_y)
                .texture(&volume_velocity_view_z)
                .texture(&volume_marker_view)
                .texture(&pressure_field_from_velocity.pressure_view())
                .texture(&pressure_field_from_density.pressure_view());
            if let Some(volume_debug_view) = volume_debug_view.as_ref() {
                bind_group_renderer_builder.texture(volume_debug_view)
            } else {
                bind_group_renderer_builder
            }
        }
        .create(device, "BindGroup: Fluid Renderers");

        // pipeline layouts.
        // Use same push constant range for all pipelines to improve internal Vulkan pipeline compatibility.
        let push_constant_ranges = &[wgpu::PushConstantRange {
            stages: wgpu::ShaderStage::COMPUTE,
            range: 0..8,
        }];

        let layout_transfer_velocity = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PipelineLayout: HybridFluid, Transfer Velocity"),
            bind_group_layouts: &[
                global_bind_group_layout,
                &group_layout_general.layout,
                &group_layout_transfer_velocity.layout,
            ],
            push_constant_ranges,
        }));
        let layout_divergence_compute = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PipelineLayout: HybridFluid, Compute Divergence"),
            bind_group_layouts: &[
                global_bind_group_layout,
                &group_layout_general.layout,
                &group_layout_divergence_compute.layout,
            ],
            push_constant_ranges,
        }));
        let layout_write_velocity_volume = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PipelineLayout: HybridFluid, Write Volume"),
            bind_group_layouts: &[
                global_bind_group_layout,
                &group_layout_general.layout,
                &group_layout_write_velocity_volume.layout,
            ],
            push_constant_ranges,
        }));
        let layout_particles = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PipelineLayout: HybridFluid, Particles"),
            bind_group_layouts: &[
                global_bind_group_layout,
                &group_layout_general.layout,
                &group_layout_advect_particles.layout,
            ],
            push_constant_ranges,
        }));

        let layout_binning = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PipelineLayout: Binning"),
            bind_group_layouts: &[global_bind_group_layout, &group_layout_general.layout, &group_layout_binning.layout],
            push_constant_ranges,
        }));

        let layout_density_projection_gather_error = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PipelineLayout: HybridFluid, Density Projection Gather"),
            bind_group_layouts: &[
                global_bind_group_layout,
                &group_layout_general.layout,
                &group_layout_density_projection_gather_error.layout,
            ],
            push_constant_ranges,
        }));
        let layout_density_projection_correct_particles = Rc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PipelineLayout: HybridFluid, Density Projection Gather"),
            bind_group_layouts: &[
                global_bind_group_layout,
                &group_layout_general.layout,
                &group_layout_density_projection_correct_particles.layout,
            ],
            push_constant_ranges,
        }));

        HybridFluid {
            grid_dimension,

            pressure_solver,
            pressure_field_from_velocity,
            pressure_field_from_density,

            volume_marker,
            volume_linked_lists,
            volume_debug,

            particles_position_llindex,
            particles_position_llindex_tmp,
            particle_binning_atomic_counter,
            simulation_properties_uniformbuffer,
            simulation_properties: SimulationPropertiesUniformBufferContent {
                num_particles: 0,
                gravity_grid: cgmath::vec3(0.0, -9.81, 0.0),
            },

            bind_group_general,
            bind_group_transfer_velocity,
            bind_group_divergence_compute,
            bind_group_divergence_projection_write_velocity,
            bind_group_advect_particles,
            bind_group_binning,
            bind_group_renderer,

            bind_group_density_projection_gather_error,
            bind_group_density_projection_correct_particles,
            bind_group_density_projection_write_velocity,

            pipeline_transfer_clear: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(
                    "Fluid: P->G, clear",
                    layout_transfer_velocity.clone(),
                    Path::new("simulation/transfer_clear.comp"),
                ),
            ),
            pipeline_transfer_build_linkedlist: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(
                    "Fluid: P->G, build linkedlists",
                    layout_transfer_velocity.clone(),
                    Path::new("simulation/transfer_build_linkedlist.comp"),
                ),
            ),
            pipeline_transfer_gather_velocity: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(
                    "Fluid: P->G, gather velocity",
                    layout_transfer_velocity.clone(),
                    Path::new("simulation/transfer_gather_velocity.comp"),
                ),
            ),
            pipeline_transfer_set_boundary_marker: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(
                    "Fluid: P->G, set boundary",
                    layout_transfer_velocity.clone(),
                    Path::new("simulation/transfer_set_boundary_marker.comp"),
                ),
            ),
            pipeline_divergence_compute: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(
                    "Fluid: Compute div",
                    layout_divergence_compute.clone(),
                    Path::new("simulation/divergence_compute.comp"),
                ),
            ),
            pipeline_divergence_remove: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(
                    "Fluid: Remove div",
                    layout_write_velocity_volume.clone(),
                    Path::new("simulation/divergence_remove.comp"),
                ),
            ),
            pipeline_extrapolate_velocity: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(
                    "Fluid: Extrapolate V",
                    layout_write_velocity_volume.clone(),
                    Path::new("simulation/extrapolate_velocity.comp"),
                ),
            ),
            pipeline_advect_particles: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(
                    "Fluid: G->P, advect",
                    layout_particles.clone(),
                    Path::new("simulation/advect_particles.comp"),
                ),
            ),

            pipeline_binning_count: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(
                    "Particle Binning: Count",
                    layout_binning.clone(),
                    Path::new("simulation/particle_binning_count.comp"),
                ),
            ),
            pipeline_binning_scan: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(
                    "Particle Binning: Scan",
                    layout_binning.clone(),
                    Path::new("simulation/particle_binning_prefixsum.comp"),
                ),
            ),
            pipeline_binning_rewrite_particles: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(
                    "Particle Binning: Rewrite particles",
                    layout_binning.clone(),
                    Path::new("simulation/particle_binning_rewrite_particles.comp"),
                ),
            ),

            pipeline_density_projection_gather_error: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(
                    "Fluid: Density Projection, gather",
                    layout_density_projection_gather_error.clone(),
                    Path::new("simulation/density_projection_gather_error.comp"),
                ),
            ),
            pipeline_density_projection_position_change: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(
                    "Fluid: Density Projection, position change",
                    layout_write_velocity_volume.clone(),
                    Path::new("simulation/density_projection_position_change.comp"),
                ),
            ),
            pipeline_density_projection_correct_particles: pipeline_manager.create_compute_pipeline(
                device,
                shader_dir,
                ComputePipelineCreationDesc::new(
                    "Fluid: Density Projection, correct",
                    layout_density_projection_correct_particles.clone(),
                    Path::new("simulation/density_projection_correct_particles.comp"),
                ),
            ),

            max_num_particles,
        }
    }

    fn clamp_to_grid(&self, grid_cor: cgmath::Point3<f32>) -> cgmath::Point3<u32> {
        // Due to the design of the grid, the 0-1 range is reserved by solid cells and can't be filled.
        // Due to the way push boundaries work, the (max-1)-max range is reserved as well!
        cgmath::Point3::new(
            (self.grid_dimension.width - 1).min(grid_cor.x as u32).max(1),
            (self.grid_dimension.height - 1).min(grid_cor.y as u32).max(1),
            (self.grid_dimension.depth_or_array_layers - 1).min(grid_cor.z as u32).max(1),
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

        self.simulation_properties.num_particles += num_new_particles;
    }

    pub fn update_signed_distance_field_for_static(
        &self,
        _device: &wgpu::Device,
        _pipeline_manager: &PipelineManager,
        _queue: &wgpu::Queue,
        _global_bind_group: &wgpu::BindGroup,
        _static_meshes: &Vec<crate::scene::models::StaticMeshData>,
        _scene_path: &Path,
    ) {
        // todo remove.
    }

    pub fn set_gravity_grid(&mut self, gravity: cgmath::Vector3<f32>) {
        self.simulation_properties.gravity_grid = gravity;
    }

    pub fn num_particles(&self) -> u32 {
        self.simulation_properties.num_particles
    }

    pub fn get_or_create_group_layout_renderer(device: &wgpu::Device) -> &BindGroupLayoutWithDesc {
        unsafe {
            GROUP_LAYOUT_RENDERER.get_or_insert_with(|| {
                let mut builder = BindGroupLayoutBuilder::new()
                    .next_binding_vertex(binding_glsl::buffer(true)) // particles, position llindex
                    .next_binding_vertex(binding_glsl::buffer(true)) // particles, velocityX
                    .next_binding_vertex(binding_glsl::buffer(true)) // particles, velocityY
                    .next_binding_vertex(binding_glsl::buffer(true)) // particles, velocityZ
                    .next_binding_vertex(binding_glsl::texture3D()) // velocityX
                    .next_binding_vertex(binding_glsl::texture3D()) // velocityY
                    .next_binding_vertex(binding_glsl::texture3D()) // velocityZ
                    .next_binding_vertex(binding_glsl::texture3D()) // marker
                    .next_binding_vertex(binding_glsl::texture3D()) // pressure
                    .next_binding_vertex(binding_glsl::texture3D()); // density
                if cfg!(debug_assertions) {
                    builder = builder.next_binding_vertex(binding_glsl::texture3D());
                }

                builder.create(device, "BindGroupLayout: ParticleRenderer")
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
        depth_or_array_layers: 8,
    };
    const COMPUTE_LOCAL_SIZE_PARTICLES: u32 = 64;
    const COMPUTE_LOCAL_SIZE_SCAN: u32 = 1024;

    pub fn pressure_solver_config_velocity(&mut self) -> &mut SolverConfig {
        &mut self.pressure_field_from_velocity.config
    }

    pub fn pressure_solver_config_density(&mut self) -> &mut SolverConfig {
        &mut self.pressure_field_from_density.config
    }

    pub fn pressure_solver_stats_velocity(&self) -> &VecDeque<SolverStatisticSample> {
        &self.pressure_field_from_velocity.stats
    }

    pub fn pressure_solver_stats_density(&self) -> &VecDeque<SolverStatisticSample> {
        &self.pressure_field_from_density.stats
    }

    // Necessary to call this to update solver statistics and config.
    // Do not call while building command buffer!
    pub fn update_statistics(&mut self) {
        self.pressure_field_from_density.start_error_buffer_readbacks();
        self.pressure_field_from_velocity.start_error_buffer_readbacks();
    }

    pub fn step(
        &mut self,
        simulation_delta: Duration,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        global_bind_group: &wgpu::BindGroup,
        pipeline_manager: &PipelineManager,
        profiler: &mut GpuProfiler,
    ) {
        wgpu_profiler!("update uniforms", profiler, encoder, device, {
            self.pressure_field_from_density.update_uniforms(queue, simulation_delta);
            self.pressure_field_from_velocity.update_uniforms(queue, simulation_delta);
            self.simulation_properties_uniformbuffer.update_content(queue, self.simulation_properties);
        });

        let grid_work_groups = wgpu_utils::compute_group_size(self.grid_dimension, Self::COMPUTE_LOCAL_SIZE_FLUID);
        let particle_work_groups = wgpu_utils::compute_group_size_1d(self.simulation_properties.num_particles, Self::COMPUTE_LOCAL_SIZE_PARTICLES);
        let scan_work_groups = wgpu_utils::compute_group_size_1d(
            self.grid_dimension.width * self.grid_dimension.height * self.grid_dimension.depth_or_array_layers,
            Self::COMPUTE_LOCAL_SIZE_SCAN,
        );

        encoder.clear_buffer(&self.particle_binning_atomic_counter, 0, None);
        if let Some(ref volume_debug) = self.volume_debug {
            encoder.clear_texture(&volume_debug, &Default::default());
        }

        wgpu_profiler!("transfer & divergence compute", profiler, encoder, device, {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("transfer & divergence compute"),
            });
            cpass.set_bind_group(0, global_bind_group, &[]);
            cpass.set_bind_group(1, &self.bind_group_general, &[]);

            wgpu_profiler!("transfer particle velocity to grid", profiler, &mut cpass, device, {
                for i in 0..3 {
                    wgpu_profiler!(&format!("dimension {}", ["x", "y", "z"][i]), profiler, &mut cpass, device, {
                        cpass.set_bind_group(2, &self.bind_group_transfer_velocity[i], &[]);
                        let scope_label = &format!("clear linked list grid{}", if i == 0 { " & marker" } else { "" });
                        wgpu_profiler!(scope_label, profiler, &mut cpass, device, {
                            cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_transfer_clear));
                            cpass.set_push_constants(0, bytemuck::bytes_of(&[i as u32]));
                            cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth_or_array_layers);
                        });

                        wgpu_profiler!("create particle linked lists", profiler, &mut cpass, device, {
                            cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_transfer_build_linkedlist));
                            cpass.dispatch(particle_work_groups, 1, 1);
                        });

                        if i == 0 {
                            wgpu_profiler!("set boundary marker", profiler, &mut cpass, device, {
                                cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_transfer_set_boundary_marker));
                                cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth_or_array_layers);
                            });
                        }

                        wgpu_profiler!("gather velocity & apply global forces", profiler, &mut cpass, device, {
                            cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_transfer_gather_velocity));
                            cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth_or_array_layers);
                        });
                    });
                }
            });

            wgpu_profiler!("compute divergence", profiler, &mut cpass, device, {
                cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_divergence_compute));
                cpass.set_bind_group(2, &self.bind_group_divergence_compute, &[]); // Writes directly into Residual of the pressure solver.
                cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth_or_array_layers);
            });
        });

        wgpu_profiler!("primary pressure solver (divergence)", profiler, encoder, device, {
            self.pressure_solver.solve(
                simulation_delta,
                encoder,
                device,
                &mut self.pressure_field_from_velocity,
                pipeline_manager,
                profiler,
            );
        });

        wgpu_profiler!("Particle Binning", profiler, encoder, device, {
            wgpu_profiler!("Clear counters", profiler, encoder, device, {
                encoder.clear_texture(&self.volume_linked_lists, &Default::default());
            });

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Particle Binning"),
                });
                cpass.set_bind_group(0, global_bind_group, &[]);
                cpass.set_bind_group(1, &self.bind_group_general, &[]);
                cpass.set_bind_group(2, &self.bind_group_binning, &[]);
                wgpu_profiler!("count", profiler, &mut cpass, device, {
                    cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_binning_count));
                    cpass.dispatch(particle_work_groups, 1, 1);
                });
                wgpu_profiler!("scan", profiler, &mut cpass, device, {
                    cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_binning_scan));
                    cpass.dispatch(scan_work_groups, 1, 1);
                });
                wgpu_profiler!("rewrite particles", profiler, &mut cpass, device, {
                    cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_binning_rewrite_particles));
                    cpass.dispatch(particle_work_groups, 1, 1);
                });
            }

            // Copy binned particles back to avoid having all descriptors twice
            wgpu_profiler!("Copy binned particles", profiler, encoder, device, {
                encoder.copy_buffer_to_buffer(
                    &self.particles_position_llindex_tmp,
                    0,
                    &self.particles_position_llindex,
                    0,
                    self.max_num_particles as u64 * std::mem::size_of::<ParticlePositionLl>() as u64,
                );
            });
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("correct for divergence / advect, compute density"),
            });
            cpass.set_bind_group(0, global_bind_group, &[]);
            cpass.set_bind_group(1, &self.bind_group_general, &[]);

            {
                cpass.set_bind_group(2, &self.bind_group_divergence_projection_write_velocity, &[]);

                wgpu_profiler!("make velocity grid divergence free", profiler, &mut cpass, device, {
                    cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_divergence_remove));
                    cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth_or_array_layers);
                });

                wgpu_profiler!("extrapolate velocity grid", profiler, &mut cpass, device, {
                    cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_extrapolate_velocity));
                    cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth_or_array_layers);
                });
            }
            wgpu_profiler!("clear marker & linked list grids", profiler, &mut cpass, device, {
                cpass.set_bind_group(2, &self.bind_group_transfer_velocity[0], &[]);
                cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_transfer_clear));
                cpass.set_push_constants(0, &bytemuck::bytes_of(&[0 as u32]));
                cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth_or_array_layers);
            });
            wgpu_profiler!("advect particles & write new linked list grid", profiler, &mut cpass, device, {
                cpass.set_bind_group(2, &self.bind_group_advect_particles, &[]);
                cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_advect_particles));
                cpass.dispatch(particle_work_groups, 1, 1);
            });

            wgpu_profiler!("density projection: set boundary marker", profiler, &mut cpass, device, {
                cpass.set_bind_group(2, &self.bind_group_transfer_velocity[0], &[]);
                cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_transfer_set_boundary_marker));
                cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth_or_array_layers);
            });
            wgpu_profiler!("density projection: compute density error via gather", profiler, &mut cpass, device, {
                cpass.set_bind_group(2, &self.bind_group_density_projection_gather_error, &[]);
                cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_density_projection_gather_error));
                cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth_or_array_layers);
            });
        }

        wgpu_profiler!("secondary pressure solver (density)", profiler, encoder, device, {
            self.pressure_solver.solve(
                simulation_delta,
                encoder,
                device,
                &mut self.pressure_field_from_density,
                pipeline_manager,
                profiler,
            );
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("correct for density error"),
            });
            cpass.set_bind_group(1, &self.bind_group_general, &[]);
            cpass.set_bind_group(0, global_bind_group, &[]);
            {
                cpass.set_bind_group(2, &self.bind_group_density_projection_write_velocity, &[]);

                wgpu_profiler!("compute position change", profiler, &mut cpass, device, {
                    cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_density_projection_position_change));
                    cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth_or_array_layers);
                });
                wgpu_profiler!("extrapolate velocity grid", profiler, &mut cpass, device, {
                    cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_extrapolate_velocity));
                    cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth_or_array_layers);
                });
            }
            wgpu_profiler!("correct particle density error", profiler, &mut cpass, device, {
                cpass.set_bind_group(2, &self.bind_group_density_projection_correct_particles, &[]);
                cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_density_projection_correct_particles));
                cpass.dispatch(particle_work_groups, 1, 1);
            });
        }
    }
}
