pub mod binding_builder;
#[allow(dead_code)]
#[allow(non_snake_case)]
pub mod binding_glsl;
pub mod gpu_profiler;
pub mod pipelines;
pub mod shader;
pub mod uniformbuffer;

pub fn compute_group_size(resource_size: wgpu::Extent3d, group_local_size: wgpu::Extent3d) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width: (resource_size.width + group_local_size.width - 1) / group_local_size.width,
        height: (resource_size.height + group_local_size.height - 1) / group_local_size.height,
        depth: (resource_size.depth + group_local_size.depth - 1) / group_local_size.depth,
    }
}

pub fn compute_group_size_1d(resource_size: u32, group_local_size: u32) -> u32 {
    (resource_size + group_local_size - 1) / group_local_size
}

macro_rules! wgpu_scope {
    ($label:expr, $profiler:expr, $encoder_or_pass:expr, $device:expr, $code:expr) => {{
        $profiler.begin_scope($label, $encoder_or_pass, $device);
        let ret = $code;
        $profiler.end_scope($encoder_or_pass);
        ret
    }};
}
