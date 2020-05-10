use notify::Watcher;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

// All entry points need to have this name.
// (could make customizable, but forcing this has perks as well)
pub const SHADER_ENTRY_POINT_NAME: &str = "main";

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
            Err(e) => error!("Failed to create filewatcher: {:?}", e),
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
    // Right now notifies any changes in the directory, if too slow consider filtering & distinguishing shaders.
    pub fn detected_change(&self) -> bool {
        self.detected_change.swap(false, Ordering::Relaxed)
    }

    pub fn load_shader_module(&self, device: &wgpu::Device, relative_path: &Path) -> Result<wgpu::ShaderModule, ()> {
        let path = self.directory.join(relative_path);

        let kind = match path.extension().and_then(OsStr::to_str) {
            Some("frag") => shaderc::ShaderKind::Fragment,
            Some("vert") => shaderc::ShaderKind::Vertex,
            Some("comp") => shaderc::ShaderKind::Compute,
            _ => {
                error!("Did not recognize file extension for shader file \"{:?}\"", path);
                return Err(());
            }
        };

        let glsl_code = match std::fs::read_to_string(&path) {
            Ok(glsl_code) => glsl_code,
            Err(err) => {
                error!("Failed to read shader file \"{:?}\": {}", path, err);
                return Err(());
            }
        };

        let spirv = {
            let mut compiler = shaderc::Compiler::new().unwrap();
            let mut options = shaderc::CompileOptions::new().unwrap();
            //options.set_hlsl_io_mapping(true);
            options.set_warnings_as_errors();
            options.set_target_env(shaderc::TargetEnv::Vulkan, 0);
            options.set_optimization_level(shaderc::OptimizationLevel::Performance);
            // options.set_optimization_level(shaderc::OptimizationLevel::Zero); // Useful for debugging/inspecting, e.g. via RenderDoc

            options.set_include_callback(|name, _ty, source_file, _depth| {
                let path = self.directory.join(name);
                match std::fs::read_to_string(&path) {
                    Ok(glsl_code) => Ok(shaderc::ResolvedInclude {
                        resolved_name: String::from(name),
                        content: glsl_code,
                    }),
                    Err(err) => Err(format!("Failed to resolve include to {} in {}: {}", name, source_file, err)),
                }
            });

            match compiler.compile_into_spirv(&glsl_code, kind, path.to_str().unwrap(), SHADER_ENTRY_POINT_NAME, Some(&options)) {
                Ok(compile_result) => {
                    if compile_result.get_num_warnings() > 0 {
                        warn!("warnings when compiling {:?}:\n{}", path, compile_result.get_warning_messages());
                    }

                    match wgpu::read_spirv(std::io::Cursor::new(&compile_result.as_binary_u8())) {
                        Ok(spirv) => spirv,
                        Err(io_error) => {
                            error!("Compilation succeeded, but wgpu::read_spirv failed: {}", io_error);
                            return Err(());
                        }
                    }
                }
                Err(compile_error) => {
                    error!("{}", compile_error);
                    return Err(());
                }
            }
        };

        // Write out the spirv shader for debugging purposes
        // {
        //     use std::io::prelude::*;
        //     let mut file = std::fs::File::create("last-shader.spv").unwrap();
        //     let data: &[u8] = unsafe { std::slice::from_raw_parts(spirv.as_ptr() as *const u8, spirv.len() * 4) };
        //     file.write_all(data).unwrap();
        // }

        Ok(device.create_shader_module(&spirv))
    }
}
