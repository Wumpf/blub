pub struct BindGroupLayoutWithDesc {
    pub layout: wgpu::BindGroupLayout,
    pub bindings: Vec<wgpu::BindGroupLayoutEntry>,
}

pub struct BindGroupLayoutBuilder {
    bindings: Vec<wgpu::BindGroupLayoutEntry>,
    next_binding_index: u32,
}

impl BindGroupLayoutBuilder {
    pub fn new() -> Self {
        BindGroupLayoutBuilder {
            bindings: Vec::new(),
            next_binding_index: 0,
        }
    }

    pub fn binding(mut self, binding: wgpu::BindGroupLayoutEntry) -> Self {
        self.next_binding_index = binding.binding + 1;
        self.bindings.push(binding);
        self
    }

    pub fn next_binding(self, visibility: wgpu::ShaderStage, ty: wgpu::BindingType) -> Self {
        let binding = self.next_binding_index;
        self.binding(wgpu::BindGroupLayoutEntry { binding, visibility, ty })
    }

    pub fn next_binding_compute(self, ty: wgpu::BindingType) -> Self {
        self.next_binding(wgpu::ShaderStage::COMPUTE, ty)
    }

    pub fn next_binding_fragment(self, ty: wgpu::BindingType) -> Self {
        self.next_binding(wgpu::ShaderStage::FRAGMENT, ty)
    }

    pub fn next_binding_vertex(self, ty: wgpu::BindingType) -> Self {
        self.next_binding(wgpu::ShaderStage::VERTEX, ty)
    }

    //pub fn next_binding_rendering(self, ty: wgpu::BindingType) -> Self {
    //    self.next_binding(wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT, ty)
    //}

    pub fn next_binding_all(self, ty: wgpu::BindingType) -> Self {
        self.next_binding(wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT | wgpu::ShaderStage::COMPUTE, ty)
    }

    pub fn create(self, device: &wgpu::Device, label: &str) -> BindGroupLayoutWithDesc {
        BindGroupLayoutWithDesc {
            layout: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                bindings: &self.bindings,
                label: Some(label),
            }),
            bindings: self.bindings,
        }
    }
}

// Builder for wgpu::BindGroups following the exact layout from a wgpu::BindGroupLayout
// Makes life simpler by assuming that order of elements in the bind group is equal to order of elements in the bind group layout.
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
    pub fn resource(mut self, resource: wgpu::BindingResource<'a>) -> Self {
        assert_lt!(self.bindings.len(), self.layout_with_desc.bindings.len());
        self.bindings.push(wgpu::Binding {
            binding: self.layout_with_desc.bindings[self.bindings.len()].binding,
            resource,
        });
        self
    }
    pub fn buffer(self, slice: wgpu::BufferSlice<'a>) -> Self {
        self.resource(wgpu::BindingResource::Buffer(slice))
    }
    pub fn sampler(self, sampler: &'a wgpu::Sampler) -> Self {
        self.resource(wgpu::BindingResource::Sampler(sampler))
    }
    pub fn texture(self, texture_view: &'a wgpu::TextureView) -> Self {
        self.resource(wgpu::BindingResource::TextureView(texture_view))
    }

    pub fn create(&self, device: &wgpu::Device, label: &str) -> wgpu::BindGroup {
        assert_eq!(self.bindings.len(), self.layout_with_desc.bindings.len());
        let descriptor = wgpu::BindGroupDescriptor {
            layout: &self.layout_with_desc.layout,
            bindings: &self.bindings,
            label: Some(label),
        };
        device.create_bind_group(&descriptor)
    }
}

// Shortcuts for resource views

pub fn simple_sampler<'a>(address_mode: wgpu::AddressMode, filter_mode: wgpu::FilterMode, label: &'a str) -> wgpu::SamplerDescriptor<'a> {
    wgpu::SamplerDescriptor {
        label: Some(label),
        address_mode_u: address_mode,
        address_mode_v: address_mode,
        address_mode_w: address_mode,
        mag_filter: filter_mode,
        min_filter: filter_mode,
        mipmap_filter: filter_mode,
        lod_min_clamp: 0.0,
        lod_max_clamp: std::f32::MAX,
        compare: wgpu::CompareFunction::Always,
    }
}
