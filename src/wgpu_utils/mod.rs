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
        depth: (resource_size.depth + group_local_size.depth - 1) / group_local_size.depth,
    }
}

pub fn compute_group_size_1d(resource_size: u32, group_local_size: u32) -> u32 {
    (resource_size + group_local_size - 1) / group_local_size
}

macro_rules! wgpu_scope {
    ($encoder_or_pass:ident, $label:expr) => {
        $encoder_or_pass.push_debug_group($label);
        #[allow(unused_mut)]
        let mut $encoder_or_pass = scopeguard::guard($encoder_or_pass, |mut encoder_or_pass| encoder_or_pass.pop_debug_group());
    };
    ($encoder_or_pass:ident, $label:expr, $code:expr) => {{
        $encoder_or_pass.push_debug_group($label);
        let ret = $code();
        $encoder_or_pass.pop_debug_group();
        ret
    }};
}
