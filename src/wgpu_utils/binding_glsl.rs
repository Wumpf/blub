// common binding types mapped to glsl type names

pub fn buffer(read_only: bool) -> wgpu::BindingType {
    wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only },
        has_dynamic_offset: false,
        min_binding_size: None,
    }
}

pub fn uniform() -> wgpu::BindingType {
    wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Uniform,
        has_dynamic_offset: false,
        min_binding_size: None,
    }
}

pub fn sampler(filtering: bool) -> wgpu::BindingType {
    wgpu::BindingType::Sampler {
        filtering,
        comparison: false,
    }
}

pub fn texture2D() -> wgpu::BindingType {
    wgpu::BindingType::Texture {
        sample_type: wgpu::TextureSampleType::Float { filterable: true },
        view_dimension: wgpu::TextureViewDimension::D2,
        multisampled: false,
    }
}

pub fn texture2DArray() -> wgpu::BindingType {
    wgpu::BindingType::Texture {
        sample_type: wgpu::TextureSampleType::Float { filterable: true },
        view_dimension: wgpu::TextureViewDimension::D2Array,
        multisampled: false,
    }
}

pub fn itexture2D() -> wgpu::BindingType {
    wgpu::BindingType::Texture {
        sample_type: wgpu::TextureSampleType::Sint,
        view_dimension: wgpu::TextureViewDimension::D2,
        multisampled: false,
    }
}

pub fn utexture2D() -> wgpu::BindingType {
    wgpu::BindingType::Texture {
        sample_type: wgpu::TextureSampleType::Uint,
        view_dimension: wgpu::TextureViewDimension::D2,
        multisampled: false,
    }
}

pub fn texture3D() -> wgpu::BindingType {
    wgpu::BindingType::Texture {
        sample_type: wgpu::TextureSampleType::Float { filterable: true },
        view_dimension: wgpu::TextureViewDimension::D3,
        multisampled: false,
    }
}

pub fn itexture3D() -> wgpu::BindingType {
    wgpu::BindingType::Texture {
        sample_type: wgpu::TextureSampleType::Sint,
        view_dimension: wgpu::TextureViewDimension::D3,
        multisampled: false,
    }
}

pub fn utexture3D() -> wgpu::BindingType {
    wgpu::BindingType::Texture {
        sample_type: wgpu::TextureSampleType::Uint,
        view_dimension: wgpu::TextureViewDimension::D3,
        multisampled: false,
    }
}

pub fn textureCube() -> wgpu::BindingType {
    wgpu::BindingType::Texture {
        sample_type: wgpu::TextureSampleType::Float { filterable: true },
        view_dimension: wgpu::TextureViewDimension::Cube,
        multisampled: false,
    }
}

pub fn image2D(format: wgpu::TextureFormat, access: wgpu::StorageTextureAccess) -> wgpu::BindingType {
    wgpu::BindingType::StorageTexture {
        access,
        view_dimension: wgpu::TextureViewDimension::D2,
        format: format,
    }
}

pub fn image2DArray(format: wgpu::TextureFormat, access: wgpu::StorageTextureAccess) -> wgpu::BindingType {
    wgpu::BindingType::StorageTexture {
        access,
        view_dimension: wgpu::TextureViewDimension::D2Array,
        format: format,
    }
}

pub fn image3D(format: wgpu::TextureFormat, access: wgpu::StorageTextureAccess) -> wgpu::BindingType {
    wgpu::BindingType::StorageTexture {
        access,
        view_dimension: wgpu::TextureViewDimension::D3,
        format: format,
    }
}
