pub fn bindingtype_storagebuffer_readonly() -> wgpu::BindingType {
    wgpu::BindingType::StorageBuffer {
        dynamic: false,
        readonly: true,
    }
}
pub fn bindingtype_storagebuffer_readwrite() -> wgpu::BindingType {
    wgpu::BindingType::StorageBuffer {
        dynamic: false,
        readonly: false,
    }
}
pub fn bindingtype_storagetexture_3d() -> wgpu::BindingType {
    wgpu::BindingType::StorageTexture {
        dimension: wgpu::TextureViewDimension::D3,
    }
}
pub fn bindingtype_texture_3d() -> wgpu::BindingType {
    wgpu::BindingType::SampledTexture {
        multisampled: false,
        dimension: wgpu::TextureViewDimension::D3,
    }
}

pub struct BindGroupLayoutWithDesc {
    pub layout: wgpu::BindGroupLayout,
    pub bindings: Vec<wgpu::BindGroupLayoutBinding>,
}

pub struct BindGroupLayoutBuilder {
    bindings: Vec<wgpu::BindGroupLayoutBinding>,
    next_binding_index: u32,
}

impl BindGroupLayoutBuilder {
    pub fn new() -> Self {
        BindGroupLayoutBuilder {
            bindings: Vec::new(),
            next_binding_index: 0,
        }
    }

    pub fn binding(mut self, binding: wgpu::BindGroupLayoutBinding) -> Self {
        self.next_binding_index = binding.binding + 1;
        self.bindings.push(binding);
        self
    }

    pub fn next_binding(self, visibility: wgpu::ShaderStage, ty: wgpu::BindingType) -> Self {
        let binding = self.next_binding_index;
        self.binding(wgpu::BindGroupLayoutBinding { binding, visibility, ty })
    }

    pub fn next_binding_compute(self, ty: wgpu::BindingType) -> Self {
        self.next_binding(wgpu::ShaderStage::COMPUTE, ty)
    }

    // pub fn next_binding_fragment(self, ty: wgpu::BindingType) -> Self {
    //     self.next_binding(wgpu::ShaderStage::FRAGMENT, ty)
    // }

    pub fn next_binding_vertex(self, ty: wgpu::BindingType) -> Self {
        self.next_binding(wgpu::ShaderStage::VERTEX, ty)
    }

    pub fn next_binding_rendering(self, ty: wgpu::BindingType) -> Self {
        self.next_binding(wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT, ty)
    }

    pub fn create(self, device: &wgpu::Device) -> BindGroupLayoutWithDesc {
        BindGroupLayoutWithDesc {
            layout: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { bindings: &self.bindings }),
            bindings: self.bindings,
        }
    }
}

pub struct BindGroupBuilder<'a> {
    layout_with_desc: &'a BindGroupLayoutWithDesc,
    bindings: Vec<wgpu::Binding<'a>>,
}

impl<'a> BindGroupBuilder<'a> {
    pub fn new(layout_with_desc: &'a BindGroupLayoutWithDesc) -> Self {
        BindGroupBuilder {
            layout_with_desc,
            bindings: Vec::new(),
        }
    }

    // Uses same binding index as binding group layout at the same ordering
    pub fn resource(&mut self, resource: wgpu::BindingResource<'a>) -> &mut Self {
        self.bindings.push(wgpu::Binding {
            binding: self.layout_with_desc.bindings[self.bindings.len()].binding,
            resource,
        });
        self
    }

    pub fn create(&self, device: &wgpu::Device) -> wgpu::BindGroup {
        let descriptor = wgpu::BindGroupDescriptor {
            layout: &self.layout_with_desc.layout,
            bindings: &self.bindings,
        };
        device.create_bind_group(&descriptor)
    }
}

// Shortcuts for resource views

pub fn default_textureview(texture_desc: &wgpu::TextureDescriptor) -> wgpu::TextureViewDescriptor {
    let dimension = match texture_desc.dimension {
        wgpu::TextureDimension::D1 => wgpu::TextureViewDimension::D1,
        wgpu::TextureDimension::D2 => {
            if texture_desc.array_layer_count > 1 {
                wgpu::TextureViewDimension::D2Array
            } else {
                wgpu::TextureViewDimension::D2
            }
        }
        wgpu::TextureDimension::D3 => wgpu::TextureViewDimension::D3,
    };

    wgpu::TextureViewDescriptor {
        format: texture_desc.format,
        dimension: dimension,
        aspect: wgpu::TextureAspect::default(),
        base_mip_level: 0,
        level_count: texture_desc.mip_level_count,
        base_array_layer: 0,
        array_layer_count: texture_desc.array_layer_count,
    }
}
