pub enum ShaderStage {
    Vertex,
    Fragment,
    #[allow(dead_code)]
    Compute,
}

pub fn load_glsl(glsl_code: &str, stage: ShaderStage) -> Option<Vec<u32>> {
    let ty = match stage {
        ShaderStage::Vertex => glsl_to_spirv::ShaderType::Vertex,
        ShaderStage::Fragment => glsl_to_spirv::ShaderType::Fragment,
        ShaderStage::Compute => glsl_to_spirv::ShaderType::Compute,
    };

    match glsl_to_spirv::compile(&glsl_code, ty) {
        Ok(compile_result) => match wgpu::read_spirv(compile_result) {
            Ok(spirv) => Some(spirv),
            Err(io_error) => {
                println!("Compilation suceeded, but wgpu::read_spirv failed: {}", io_error);
                None
            }
        },
        Err(compile_error) => {
            println!("{}", compile_error);
            None
        }
    }
}

pub fn create_glsl_shader_module(device: &wgpu::Device, glsl_code: &str, stage: ShaderStage) -> Option<wgpu::ShaderModule> {
    match &load_glsl(glsl_code, stage) {
        Some(spirv) => Some(device.create_shader_module(spirv)),
        None => None,
    }
}
