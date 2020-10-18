// common binding types mapped to glsl type names

pub fn buffer(readonly: bool) -> wgpu::BindingType {
    wgpu::BindingType::StorageBuffer {
        dynamic: false,
        min_binding_size: None,
        readonly,
    }
}

pub fn uniform() -> wgpu::BindingType {
    wgpu::BindingType::UniformBuffer {
        dynamic: false,
        min_binding_size: None,
    }
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

pub fn texture2DArray() -> wgpu::BindingType {
    wgpu::BindingType::SampledTexture {
        multisampled: false,
        component_type: wgpu::TextureComponentType::Float,
        dimension: wgpu::TextureViewDimension::D2Array,
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

pub fn textureCube() -> wgpu::BindingType {
    wgpu::BindingType::SampledTexture {
        multisampled: false,
        component_type: wgpu::TextureComponentType::Float,
        dimension: wgpu::TextureViewDimension::Cube,
    }
}

pub fn image2D(format: wgpu::TextureFormat, readonly: bool) -> wgpu::BindingType {
    wgpu::BindingType::StorageTexture {
        dimension: wgpu::TextureViewDimension::D2,
        format: format,
        readonly,
    }
}

pub fn image2DArray(format: wgpu::TextureFormat, readonly: bool) -> wgpu::BindingType {
    wgpu::BindingType::StorageTexture {
        dimension: wgpu::TextureViewDimension::D2Array,
        format: format,
        readonly,
    }
}

pub fn iimage2D(format: wgpu::TextureFormat, readonly: bool) -> wgpu::BindingType {
    wgpu::BindingType::StorageTexture {
        dimension: wgpu::TextureViewDimension::D2,
        format: format,
        readonly,
    }
}

pub fn uimage2D(format: wgpu::TextureFormat, readonly: bool) -> wgpu::BindingType {
    wgpu::BindingType::StorageTexture {
        dimension: wgpu::TextureViewDimension::D2,
        format: format,
        readonly,
    }
}

pub fn image3D(format: wgpu::TextureFormat, readonly: bool) -> wgpu::BindingType {
    wgpu::BindingType::StorageTexture {
        dimension: wgpu::TextureViewDimension::D3,
        format: format,
        readonly,
    }
}

pub fn iimage3D(format: wgpu::TextureFormat, readonly: bool) -> wgpu::BindingType {
    wgpu::BindingType::StorageTexture {
        dimension: wgpu::TextureViewDimension::D3,
        format: format,
        readonly,
    }
}

pub fn uimage3D(format: wgpu::TextureFormat, readonly: bool) -> wgpu::BindingType {
    wgpu::BindingType::StorageTexture {
        dimension: wgpu::TextureViewDimension::D3,
        format: format,
        readonly,
    }
}
