use crate::{
    global_ubo::GlobalUBO,
    scene::models::SceneModels,
    wgpu_utils::{binding_builder::*, binding_glsl},
};

pub struct GlobalBindings {
    bind_group_layout: BindGroupLayoutWithDesc,
    bind_group: Option<wgpu::BindGroup>,
}

impl GlobalBindings {
    pub const NUM_MESH_TEXTURES: u32 = 1;

    pub fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = BindGroupLayoutBuilder::new()
            // Constants
            .next_binding_all(binding_glsl::uniform())
            // Sampler
            .next_binding_all(binding_glsl::sampler(true))
            .next_binding_all(binding_glsl::sampler(false))
            // Meshdata
            .next_binding(
                wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT | wgpu::ShaderStage::COMPUTE,
                binding_glsl::buffer(true),
            )
            .binding(wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStage::FRAGMENT,
                ty: binding_glsl::texture2D(),
                count: std::num::NonZeroU32::new(Self::NUM_MESH_TEXTURES),
            })
            .next_binding(wgpu::ShaderStage::COMPUTE | wgpu::ShaderStage::VERTEX, binding_glsl::buffer(true))
            .next_binding(wgpu::ShaderStage::COMPUTE | wgpu::ShaderStage::VERTEX, binding_glsl::buffer(true))
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

        let dummy_texture_view = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("Dummy Texture"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsage::SAMPLED,
            })
            .create_view(&Default::default());

        let texture_views: Vec<&wgpu::TextureView> = meshes
            .texture_views
            .iter()
            .chain(std::iter::repeat(&dummy_texture_view).take(Self::NUM_MESH_TEXTURES as usize - meshes.texture_views.len()))
            .collect();

        self.bind_group = Some(
            BindGroupBuilder::new(&self.bind_group_layout)
                // Constants
                .resource(ubo.binding_resource())
                // Sampler
                .sampler(&trilinear_sampler)
                .sampler(&point_sampler)
                // Meshdata
                .resource(meshes.mesh_desc_buffer.as_entire_binding())
                .resource(wgpu::BindingResource::TextureViewArray(&texture_views))
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
