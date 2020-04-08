// common binding types mapped to glsl type names

pub fn buffer(readonly: bool) -> wgpu::BindingType {
    wgpu::BindingType::StorageBuffer { dynamic: false, readonly }
}

pub fn uniform() -> wgpu::BindingType {
    wgpu::BindingType::UniformBuffer { dynamic: false }
}

pub fn sampler() -> wgpu::BindingType {
    wgpu::BindingType::Sampler { comparison: false }
}

pub fn texture2D() -> wgpu::BindingType {
    wgpu::BindingType::SampledTexture {
        multisampled: false,
        component_type: wgpu::TextureComponentType::Float,
        dimension: wgpu::TextureViewDimension::D2,
    }
}

pub fn itexture2D() -> wgpu::BindingType {
    wgpu::BindingType::SampledTexture {
        multisampled: false,
        component_type: wgpu::TextureComponentType::Sint,
        dimension: wgpu::TextureViewDimension::D2,
    }
}

pub fn utexture2D() -> wgpu::BindingType {
    wgpu::BindingType::SampledTexture {
        multisampled: false,
        component_type: wgpu::TextureComponentType::Uint,
        dimension: wgpu::TextureViewDimension::D2,
    }
}

pub fn texture3D() -> wgpu::BindingType {
    wgpu::BindingType::SampledTexture {
        multisampled: false,
        component_type: wgpu::TextureComponentType::Float,
        dimension: wgpu::TextureViewDimension::D3,
    }
}

pub fn itexture3D() -> wgpu::BindingType {
    wgpu::BindingType::SampledTexture {
        multisampled: false,
        component_type: wgpu::TextureComponentType::Sint,
        dimension: wgpu::TextureViewDimension::D3,
    }
}

pub fn utexture3D() -> wgpu::BindingType {
    wgpu::BindingType::SampledTexture {
        multisampled: false,
        component_type: wgpu::TextureComponentType::Uint,
        dimension: wgpu::TextureViewDimension::D3,
    }
}

pub fn image2d(format: wgpu::TextureFormat, readonly: bool) -> wgpu::BindingType {
    wgpu::BindingType::StorageTexture {
        dimension: wgpu::TextureViewDimension::D2,
        component_type: wgpu::TextureComponentType::Float,
        format: format,
        readonly,
    }
}

pub fn iimage2d(format: wgpu::TextureFormat, readonly: bool) -> wgpu::BindingType {
    wgpu::BindingType::StorageTexture {
        dimension: wgpu::TextureViewDimension::D2,
        component_type: wgpu::TextureComponentType::Sint,
        format: format,
        readonly,
    }
}

pub fn uimage2d(format: wgpu::TextureFormat, readonly: bool) -> wgpu::BindingType {
    wgpu::BindingType::StorageTexture {
        dimension: wgpu::TextureViewDimension::D2,
        component_type: wgpu::TextureComponentType::Uint,
        format: format,
        readonly,
    }
}

pub fn image3d(format: wgpu::TextureFormat, readonly: bool) -> wgpu::BindingType {
    wgpu::BindingType::StorageTexture {
        dimension: wgpu::TextureViewDimension::D3,
        component_type: wgpu::TextureComponentType::Float,
        format: format,
        readonly,
    }
}

pub fn iimage3d(format: wgpu::TextureFormat, readonly: bool) -> wgpu::BindingType {
    wgpu::BindingType::StorageTexture {
        dimension: wgpu::TextureViewDimension::D3,
        component_type: wgpu::TextureComponentType::Sint,
        format: format,
        readonly,
    }
}

pub fn uimage3d(format: wgpu::TextureFormat, readonly: bool) -> wgpu::BindingType {
    wgpu::BindingType::StorageTexture {
        dimension: wgpu::TextureViewDimension::D3,
        component_type: wgpu::TextureComponentType::Uint,
        format: format,
        readonly,
    }
}
