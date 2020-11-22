use crate::wgpu_utils::*;
use crate::{global_ubo::GlobalUBO, scene_models::SceneModels, wgpu_utils::binding_builder::*};

pub struct GlobalBindings {
    bind_group_layout: BindGroupLayoutWithDesc,
    bind_group: Option<wgpu::BindGroup>,
}

impl GlobalBindings {
    pub fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = BindGroupLayoutBuilder::new()
            // Constants
            .next_binding_all(binding_glsl::uniform())
            // Sampler
            .next_binding_all(binding_glsl::sampler())
            .next_binding_all(binding_glsl::sampler())
            // Meshdata
            .next_binding(
                wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT | wgpu::ShaderStage::COMPUTE,
                binding_glsl::buffer(true),
            )
            .next_binding(wgpu::ShaderStage::COMPUTE, binding_glsl::buffer(true)) // Index buffer used for compute shader consuming the mesh
            .next_binding(wgpu::ShaderStage::COMPUTE, binding_glsl::buffer(true)) // Vertex buffer used for compute shader consuming the mesh
            .create(device, "BindGroupLayout: GlobalBindings");

        GlobalBindings {
            bind_group_layout: bind_group_layout,
            bind_group: None,
        }
    }

    pub fn create_bind_group(&mut self, device: &wgpu::Device, ubo: &GlobalUBO, meshes: &SceneModels) {
        let trilinear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Sampler LinearClamp (global)"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let point_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Sampler NearestClamp (global)"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        self.bind_group = Some(
            BindGroupBuilder::new(&self.bind_group_layout)
                // Constants
                .resource(ubo.binding_resource())
                // Sampler
                .sampler(&trilinear_sampler)
                .sampler(&point_sampler)
                // Meshdata
                .resource(meshes.mesh_desc_buffer.as_entire_binding())
                .resource(meshes.index_buffer.as_entire_binding())
                .resource(meshes.vertex_buffer.as_entire_binding())
                .create(device, "BindGroup: GlobalBindings"),
        );
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group.as_ref().expect("Bind group has not been created yet!")
    }

    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout.layout
    }
}
