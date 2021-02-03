use super::{pressure_solver::*, signed_distance_field::SignedDistanceField};
use crate::wgpu_utils::{self, binding_builder::*, binding_glsl, pipelines::*, shader::*, uniformbuffer::*};
use rand::prelude::*;
use std::{collections::VecDeque, path::Path, rc::Rc, time::Duration};

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

    signed_distance_field: SignedDistanceField,

    particles_position_llindex: wgpu::Buffer,
    simulation_properties_uniformbuffer: UniformBuffer<SimulationPropertiesUniformBufferContent>,
    simulation_properties: SimulationPropertiesUniformBufferContent,

    bind_group_general: wgpu::BindGroup,
    bind_group_transfer_velocity: [wgpu::BindGroup; 3],
    bind_group_divergence_compute: wgpu::BindGroup,
    bind_group_divergence_projection_write_velocity: wgpu::BindGroup,
    bind_group_advect_particles: wgpu::BindGroup,
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
        let volume_marker_primary = device.create_texture(&create_volume_texture_desc("Marker Grid", wgpu::TextureFormat::R8Snorm));
        // TODO: Reuse (a) pressure volume for this. (pressure is only defined at FLUID, this is only defined at SOLID)
        let volume_penetration_depth = device.create_texture(&create_volume_texture_desc("Solid Penetration Depth", wgpu::TextureFormat::R32Uint));
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
        let volume_marker_view = volume_marker_primary.create_view(&Default::default());
        let volume_penetration_depth_view = volume_penetration_depth.create_view(&Default::default());
        let volume_debug_view = match volume_debug {
            Some(volume) => Some(volume.create_view(&Default::default())),
            None => None,
        };

        // Layouts
        let group_layout_general = if volume_debug_view.is_some() {
            BindGroupLayoutBuilder::new()
                .next_binding_compute(binding_glsl::uniform())
                // Debug Volume
                .next_binding_compute(binding_glsl::image3D(
                    wgpu::TextureFormat::R32Float,
                    wgpu::StorageTextureAccess::ReadWrite,
                ))
        } else {
            BindGroupLayoutBuilder::new().next_binding_compute(binding_glsl::uniform())
        }
        .create(device, "BindGroupLayout: HybridFluid Uniform");
        let group_layout_transfer_velocity = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::buffer(false)) // particles, position llindex
            .next_binding_compute(binding_glsl::buffer(true)) // particles, velocity component
            .next_binding_compute(binding_glsl::uimage3D(
                wgpu::TextureFormat::R32Uint,
                wgpu::StorageTextureAccess::ReadWrite,
            )) // linkedlist_volume
            .next_binding_compute(binding_glsl::image3D(wgpu::TextureFormat::R8Snorm, wgpu::StorageTextureAccess::ReadWrite)) // marker volume
            .next_binding_compute(binding_glsl::image3D(
                wgpu::TextureFormat::R32Float,
                wgpu::StorageTextureAccess::ReadWrite,
            )) // velocity component
            .next_binding_compute(binding_glsl::texture3D()) // marker for static objects
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
            .next_binding_compute(binding_glsl::image3D(wgpu::TextureFormat::R32Uint, wgpu::StorageTextureAccess::ReadWrite)) // penetration depth / unused
            .create(device, "BindGroupLayout: Write to Velocity");
        let group_layout_advect_particles = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::texture3D()) // velocityX
            .next_binding_compute(binding_glsl::texture3D()) // velocityY
            .next_binding_compute(binding_glsl::texture3D()) // velocityZ
            .next_binding_compute(binding_glsl::image3D(wgpu::TextureFormat::R8Snorm, wgpu::StorageTextureAccess::ReadWrite)) // marker volume
            .next_binding_compute(binding_glsl::uimage3D(
                wgpu::TextureFormat::R32Uint,
                wgpu::StorageTextureAccess::ReadWrite,
            )) // linkedlist_volume
            .next_binding_compute(binding_glsl::buffer(false)) // particles, position llindex
            .next_binding_compute(binding_glsl::buffer(false)) // particles, velocityX
            .next_binding_compute(binding_glsl::buffer(false)) // particles, velocityY
            .next_binding_compute(binding_glsl::buffer(false)) // particles, velocityZ
            .next_binding_compute(binding_glsl::uimage3D(
                wgpu::TextureFormat::R32Uint,
                wgpu::StorageTextureAccess::ReadWrite,
            )) // penetration depth
            .next_binding_compute(binding_glsl::texture3D()) // Distance field
            .create(device, "BindGroupLayout: Advect to Particles");

        let group_layout_density_projection_gather_error = BindGroupLayoutBuilder::new()
            .next_binding_compute(binding_glsl::buffer(false)) // particles, position llindex
            .next_binding_compute(binding_glsl::utexture3D()) // linkedlist_volume
            .next_binding_compute(binding_glsl::image3D(wgpu::TextureFormat::R8Snorm, wgpu::StorageTextureAccess::ReadWrite)) // marker volume
            .next_binding_compute(binding_glsl::image3D(
                wgpu::TextureFormat::R32Float,
                wgpu::StorageTextureAccess::ReadWrite,
            )) // density volume
            .next_binding_compute(binding_glsl::utexture3D()) // penetration depth
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

        let signed_distance_field = SignedDistanceField::new(device, grid_dimension, shader_dir, pipeline_manager, global_bind_group_layout);

        // Bind groups.
        let bind_group_general = match volume_debug_view.as_ref() {
            Some(volume_debug_view) => BindGroupBuilder::new(&group_layout_general)
                .resource(simulation_properties_uniformbuffer.binding_resource())
                .texture(volume_debug_view),
            None => BindGroupBuilder::new(&group_layout_general).resource(simulation_properties_uniformbuffer.binding_resource()),
        }
        .create(device, "BindGroup: HybridFluid Uniform");

        let bind_group_transfer_velocity = [
            BindGroupBuilder::new(&group_layout_transfer_velocity)
                .resource(particles_position_llindex.as_entire_binding())
                .resource(particles_velocity_x.as_entire_binding())
                .texture(&volume_linked_lists_view)
                .texture(&volume_marker_view)
                .texture(&volume_velocity_view_x)
                .texture(signed_distance_field.texture_view())
                .create(device, "BindGroup: Transfer velocity to volume X"),
            BindGroupBuilder::new(&group_layout_transfer_velocity)
                .resource(particles_position_llindex.as_entire_binding())
                .resource(particles_velocity_y.as_entire_binding())
                .texture(&volume_linked_lists_view)
                .texture(&volume_marker_view)
                .texture(&volume_velocity_view_y)
                .texture(signed_distance_field.texture_view())
                .create(device, "BindGroup: Transfer velocity to volume Y"),
            BindGroupBuilder::new(&group_layout_transfer_velocity)
                .resource(particles_position_llindex.as_entire_binding())
                .resource(particles_velocity_z.as_entire_binding())
                .texture(&volume_linked_lists_view)
                .texture(&volume_marker_view)
                .texture(&volume_velocity_view_z)
                .texture(signed_distance_field.texture_view())
                .create(device, "BindGroup: Transfer velocity to volume Z"),
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
            .texture(&volume_penetration_depth_view) // TODO: Not great to have this here
            .create(device, "BindGroup: Write to Velocity Grid - divergence projection");
        let bind_group_density_projection_write_velocity = BindGroupBuilder::new(&group_layout_write_velocity_volume)
            .texture(&volume_marker_view)
            .texture(&volume_velocity_view_x)
            .texture(&volume_velocity_view_y)
            .texture(&volume_velocity_view_z)
            .texture(pressure_field_from_density.pressure_view())
            .texture(&volume_penetration_depth_view)
            .create(device, "BindGroup: Write to Velocity Grid - density projection");
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
            .texture(&volume_penetration_depth_view)
            .texture(signed_distance_field.texture_view())
            .create(device, "BindGroup: Write to Particles");
        let bind_group_density_projection_gather_error = BindGroupBuilder::new(&group_layout_density_projection_gather_error)
            .resource(particles_position_llindex.as_entire_binding())
            .texture(&volume_linked_lists_view)
            .texture(&volume_marker_view)
            .texture(&pressure_solver.residual_view())
            .texture(&volume_penetration_depth_view)
            .create(device, "BindGroup: Density projection gather");
        let bind_group_density_projection_correct_particles = BindGroupBuilder::new(&group_layout_density_projection_correct_particles)
            .resource(particles_position_llindex.as_entire_binding())
            .texture(&volume_marker_view)
            .texture(&volume_velocity_view_x)
            .texture(&volume_velocity_view_y)
            .texture(&volume_velocity_view_z)
            .create(device, "BindGroup: Density projection correct particles");

        let mut bind_group_renderer_builder = BindGroupBuilder::new(&Self::get_or_create_group_layout_renderer(device))
            .resource(particles_position_llindex.as_entire_binding())
            .resource(particles_velocity_x.as_entire_binding())
            .resource(particles_velocity_y.as_entire_binding())
            .resource(particles_velocity_z.as_entire_binding())
            .texture(&volume_velocity_view_x)
            .texture(&volume_velocity_view_y)
            .texture(&volume_velocity_view_z)
            .texture(&volume_marker_view)
            .texture(&pressure_field_from_velocity.pressure_view())
            .texture(&pressure_field_from_density.pressure_view())
            .texture(signed_distance_field.texture_view());
        if let Some(volume_debug_view) = volume_debug_view.as_ref() {
            bind_group_renderer_builder = bind_group_renderer_builder.texture(volume_debug_view);
        }
        let bind_group_renderer = bind_group_renderer_builder.create(device, "BindGroup: Fluid Renderers");

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
            bind_group_layouts: &[global_bind_group_layout, &group_layout_divergence_compute.layout],
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

            signed_distance_field,

            particles_position_llindex,
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
            (self.grid_dimension.depth - 1).min(grid_cor.z as u32).max(1),
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
        device: &wgpu::Device,
        pipeline_manager: &PipelineManager,
        queue: &wgpu::Queue,
        global_bind_group: &wgpu::BindGroup,
        static_meshes: &Vec<crate::scene_models::MeshData>,
        scene_path: &Path,
    ) {
        let cache_filename = scene_path.parent().unwrap().join(format!(
            ".{}.static_signed_distance_field.cache",
            scene_path.file_name().unwrap().to_str().unwrap()
        ));
        match self.signed_distance_field.load_signed_distance_field(&cache_filename, queue) {
            Ok(_) => {}
            Err(_) => {
                self.signed_distance_field
                    .compute_distance_field_for_static(device, pipeline_manager, queue, global_bind_group, static_meshes);
                self.signed_distance_field.save(&cache_filename, device, queue);
            }
        }
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
                    .next_binding_vertex(binding_glsl::texture3D()) // density
                    .next_binding_vertex(binding_glsl::texture3D()); // distance field
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
        depth: 8,
    };
    const COMPUTE_LOCAL_SIZE_PARTICLES: u32 = 64;

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
        pipeline_manager: &PipelineManager,
        queue: &wgpu::Queue,
        global_bind_group: &wgpu::BindGroup,
    ) {
        wgpu_scope!(encoder, "HybridFluid.step");

        wgpu_scope!(encoder, "update uniforms", || {
            self.pressure_field_from_density.update_uniforms(queue, simulation_delta);
            self.pressure_field_from_velocity.update_uniforms(queue, simulation_delta);
            self.simulation_properties_uniformbuffer.update_content(queue, self.simulation_properties);
        });

        let grid_work_groups = wgpu_utils::compute_group_size(self.grid_dimension, Self::COMPUTE_LOCAL_SIZE_FLUID);
        let particle_work_groups = wgpu_utils::compute_group_size_1d(self.simulation_properties.num_particles, Self::COMPUTE_LOCAL_SIZE_PARTICLES);

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("transfer & divergence compute"),
            });
            cpass.set_bind_group(0, global_bind_group, &[]);
            cpass.set_bind_group(1, &self.bind_group_general, &[]);

            wgpu_scope!(cpass, "transfer particle velocity to grid", || {
                for i in 0..3 {
                    wgpu_scope!(cpass, &format!("dimension {}", ["x", "y", "z"][i]), || {
                        cpass.set_bind_group(2, &self.bind_group_transfer_velocity[i], &[]);
                        wgpu_scope!(cpass, &format!("clear linked list grid{}", if i == 0 { " & marker" } else { "" }), || {
                            cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_transfer_clear));
                            cpass.set_push_constants(0, bytemuck::bytes_of(&[i as u32]));
                            cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);
                        });

                        wgpu_scope!(cpass, "create particle linked lists", || {
                            cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_transfer_build_linkedlist));
                            cpass.dispatch(particle_work_groups, 1, 1);
                        });

                        if i == 0 {
                            wgpu_scope!(cpass, "set boundary marker", || {
                                cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_transfer_set_boundary_marker));
                                cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);
                            });
                        }

                        wgpu_scope!(cpass, "gather velocity & apply global forces", || {
                            cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_transfer_gather_velocity));
                            cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);
                        });
                    });
                }
            });
            wgpu_scope!(cpass, "compute divergence", || {
                cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_divergence_compute));
                cpass.set_bind_group(1, &self.bind_group_divergence_compute, &[]); // Writes directly into Residual of the pressure solver.
                cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);
            });
        }

        self.pressure_solver
            .solve(simulation_delta, &mut self.pressure_field_from_velocity, &mut encoder, pipeline_manager);

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("correct for divergence / advect, compute density"),
            });
            cpass.set_bind_group(0, global_bind_group, &[]);
            cpass.set_bind_group(1, &self.bind_group_general, &[]);

            {
                cpass.set_bind_group(2, &self.bind_group_divergence_projection_write_velocity, &[]);

                wgpu_scope!(cpass, "make velocity grid divergence free", || {
                    cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_divergence_remove));
                    cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);
                });

                wgpu_scope!(cpass, "extrapolate velocity grid", || {
                    cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_extrapolate_velocity));
                    cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);
                });
            }
            wgpu_scope!(cpass, "clear marker & linked list grids", || {
                cpass.set_bind_group(2, &self.bind_group_transfer_velocity[0], &[]);
                cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_transfer_clear));
                cpass.set_push_constants(0, &bytemuck::bytes_of(&[0 as u32]));
                cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);
            });
            wgpu_scope!(cpass, "advect particles & write new linked list grid", || {
                cpass.set_bind_group(2, &self.bind_group_advect_particles, &[]);
                cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_advect_particles));
                cpass.dispatch(particle_work_groups, 1, 1);
            });

            wgpu_scope!(cpass, "density projection: set boundary marker", || {
                cpass.set_bind_group(2, &self.bind_group_transfer_velocity[0], &[]);
                cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_transfer_set_boundary_marker));
                cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);
            });
            wgpu_scope!(cpass, "density projection: compute density error via gather", || {
                cpass.set_bind_group(2, &self.bind_group_density_projection_gather_error, &[]);
                cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_density_projection_gather_error));
                cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);
            });
        }

        // Compute pressure from density error.
        self.pressure_solver
            .solve(simulation_delta, &mut self.pressure_field_from_density, &mut encoder, pipeline_manager);

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("correct for density error"),
            });
            cpass.set_bind_group(1, &self.bind_group_general, &[]);
            cpass.set_bind_group(0, global_bind_group, &[]);
            {
                cpass.set_bind_group(2, &self.bind_group_density_projection_write_velocity, &[]);

                wgpu_scope!(cpass, "compute position change", || {
                    cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_density_projection_position_change));
                    cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);
                });
                wgpu_scope!(cpass, "extrapolate velocity grid", || {
                    cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_extrapolate_velocity));
                    cpass.dispatch(grid_work_groups.width, grid_work_groups.height, grid_work_groups.depth);
                });
            }
            wgpu_scope!(cpass, "correct particle density error", || {
                cpass.set_bind_group(2, &self.bind_group_density_projection_correct_particles, &[]);
                cpass.set_pipeline(pipeline_manager.get_compute(&self.pipeline_density_projection_correct_particles));
                cpass.dispatch(particle_work_groups, 1, 1);
            });
        }
    }
}
