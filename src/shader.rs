use notify::Watcher;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub enum ShaderStage {
    Vertex,
    Fragment,
    #[allow(dead_code)]
    Compute,
}

fn load_shader(glsl_code: &str, stage: ShaderStage) -> Option<Vec<u32>> {
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

fn load_shader_module(device: &wgpu::Device, glsl_code: &str, stage: ShaderStage) -> Option<wgpu::ShaderModule> {
    match &load_shader(glsl_code, stage) {
        Some(spirv) => Some(device.create_shader_module(spirv)),
        None => None,
    }
}

pub struct ShaderDirectory {
    #[allow(dead_code)]
    watcher: notify::RecommendedWatcher,
    detected_change: Arc<AtomicBool>,
    directory: PathBuf,
}

impl ShaderDirectory {
    pub fn new(path: &Path) -> ShaderDirectory {
        let detected_change = Arc::new(AtomicBool::new(false));
        let detected_change_evt_ref = detected_change.clone();
        let mut watcher: notify::RecommendedWatcher = notify::Watcher::new_immediate(move |res| match res {
            Ok(_) => detected_change_evt_ref.store(true, Ordering::Relaxed),
            Err(e) => println!("Failed to create filewatcher: {:?}", e),
        })
        .unwrap();
        watcher.watch(path, notify::RecursiveMode::Recursive).unwrap();

        ShaderDirectory {
            watcher,
            detected_change,
            directory: PathBuf::from(path),
        }
    }

    // Checks if any change was detected in the shader directory.
    // Right now notifies any all changes in the directory, if too slow consider filtering & distinguishing shaders.
    pub fn detected_change(&self) -> bool {
        self.detected_change.swap(false, Ordering::Relaxed)
    }

    pub fn load_shader_module(&self, device: &wgpu::Device, relative_filename: &Path) -> Option<wgpu::ShaderModule> {
        let path = self.directory.join(relative_filename);

        let shader_stage = match path.extension().and_then(OsStr::to_str) {
            Some("frag") => ShaderStage::Fragment,
            Some("vert") => ShaderStage::Vertex,
            _ => {
                println!("Did not recognize file extension for shader file \"{:?}\"", path);
                return None;
            }
        };

        match std::fs::read_to_string(&path) {
            Ok(glsl_code) => load_shader_module(device, &glsl_code, shader_stage),
            Err(err) => {
                println!("Failed to read shader file \"{:?}\": {}", path, err);
                None
            }
        }
    }
}
