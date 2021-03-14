pub mod binding_builder;
#[allow(dead_code)]
#[allow(non_snake_case)]
pub mod binding_glsl;
pub mod pipelines;
pub mod shader;
pub mod uniformbuffer;

pub fn compute_group_size(resource_size: wgpu::Extent3d, group_local_size: wgpu::Extent3d) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width: (resource_size.width + group_local_size.width - 1) / group_local_size.width,
        height: (resource_size.height + group_local_size.height - 1) / group_local_size.height,
        depth_or_array_layers: (resource_size.depth_or_array_layers + group_local_size.depth_or_array_layers - 1)
            / group_local_size.depth_or_array_layers,
    }
}

pub fn compute_group_size_1d(resource_size: u32, group_local_size: u32) -> u32 {
    (resource_size + group_local_size - 1) / group_local_size
}
