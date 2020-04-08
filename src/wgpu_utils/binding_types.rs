// common binding types mapped to glsl type names

pub fn bindingtype_buffer(readonly: bool) -> wgpu::BindingType {
    wgpu::BindingType::StorageBuffer { dynamic: false, readonly }
}

pub fn bindingtype_uniform() -> wgpu::BindingType {
    wgpu::BindingType::UniformBuffer { dynamic: false }
}

pub fn bindingtype_sampler() -> wgpu::BindingType {
    wgpu::BindingType::Sampler { comparison: false }
}

pub fn bindingtype_texture2D() -> wgpu::BindingType {
    wgpu::BindingType::SampledTexture {
        multisampled: false,
        component_type: wgpu::TextureComponentType::Float,
        dimension: wgpu::TextureViewDimension::D2,
    }
}

pub fn bindingtype_itexture2D() -> wgpu::BindingType {
    wgpu::BindingType::SampledTexture {
        multisampled: false,
        component_type: wgpu::TextureComponentType::Sint,
        dimension: wgpu::TextureViewDimension::D2,
    }
}

pub fn bindingtype_utexture2D() -> wgpu::BindingType {
    wgpu::BindingType::SampledTexture {
        multisampled: false,
        component_type: wgpu::TextureComponentType::Uint,
        dimension: wgpu::TextureViewDimension::D2,
    }
}

pub fn bindingtype_texture3D() -> wgpu::BindingType {
    wgpu::BindingType::SampledTexture {
        multisampled: false,
        component_type: wgpu::TextureComponentType::Float,
        dimension: wgpu::TextureViewDimension::D3,
    }
}

pub fn bindingtype_itexture3D() -> wgpu::BindingType {
    wgpu::BindingType::SampledTexture {
        multisampled: false,
        component_type: wgpu::TextureComponentType::Sint,
        dimension: wgpu::TextureViewDimension::D3,
    }
}

pub fn bindingtype_utexture3D() -> wgpu::BindingType {
    wgpu::BindingType::SampledTexture {
        multisampled: false,
        component_type: wgpu::TextureComponentType::Uint,
        dimension: wgpu::TextureViewDimension::D3,
    }
}

pub fn bindingtype_image2d(format: wgpu::TextureFormat, readonly: bool) -> wgpu::BindingType {
    wgpu::BindingType::StorageTexture {
        dimension: wgpu::TextureViewDimension::D2,
        component_type: wgpu::TextureComponentType::Float,
        format: format,
        readonly,
    }
}

pub fn bindingtype_iimage2d(format: wgpu::TextureFormat, readonly: bool) -> wgpu::BindingType {
    wgpu::BindingType::StorageTexture {
        dimension: wgpu::TextureViewDimension::D2,
        component_type: wgpu::TextureComponentType::Sint,
        format: format,
        readonly,
    }
}

pub fn bindingtype_uimage2d(format: wgpu::TextureFormat, readonly: bool) -> wgpu::BindingType {
    wgpu::BindingType::StorageTexture {
        dimension: wgpu::TextureViewDimension::D2,
        component_type: wgpu::TextureComponentType::Uint,
        format: format,
        readonly,
    }
}

pub fn bindingtype_image3d(format: wgpu::TextureFormat, readonly: bool) -> wgpu::BindingType {
    wgpu::BindingType::StorageTexture {
        dimension: wgpu::TextureViewDimension::D3,
        component_type: wgpu::TextureComponentType::Float,
        format: format,
        readonly,
    }
}

pub fn bindingtype_iimage3d(format: wgpu::TextureFormat, readonly: bool) -> wgpu::BindingType {
    wgpu::BindingType::StorageTexture {
        dimension: wgpu::TextureViewDimension::D3,
        component_type: wgpu::TextureComponentType::Sint,
        format: format,
        readonly,
    }
}

pub fn bindingtype_uimage3d(format: wgpu::TextureFormat, readonly: bool) -> wgpu::BindingType {
    wgpu::BindingType::StorageTexture {
        dimension: wgpu::TextureViewDimension::D3,
        component_type: wgpu::TextureComponentType::Uint,
        format: format,
        readonly,
    }
}
