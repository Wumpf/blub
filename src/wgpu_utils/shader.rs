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

        let compilation_artifact = {
            let mut compiler = shaderc::Compiler::new().unwrap();
            let mut options = shaderc::CompileOptions::new().unwrap();
            //options.set_hlsl_io_mapping(true);
            options.set_warnings_as_errors();
            options.set_target_env(shaderc::TargetEnv::Vulkan, 0);
            if cfg!(debug_assertions) {
                options.set_optimization_level(shaderc::OptimizationLevel::Zero);
            } else {
                options.set_optimization_level(shaderc::OptimizationLevel::Performance);
            }
            // Helps a lot when inspecting in ShaderDoc (will show all original source files before processing) but doesn't seem to hurt performance at all :)
            options.set_generate_debug_info();

            options.set_include_callback(|name, include_type, source_file, _depth| {
                let path = if include_type == shaderc::IncludeType::Relative {
                    Path::new(Path::new(source_file).parent().unwrap()).join(name)
                } else {
                    self.directory.join(name)
                };
                match std::fs::read_to_string(&path) {
                    Ok(glsl_code) => Ok(shaderc::ResolvedInclude {
                        resolved_name: String::from(name),
                        content: glsl_code,
                    }),
                    Err(err) => Err(format!(
                        "Failed to resolve include to {} in {} (was looking for {:?}): {}",
                        name, source_file, path, err
                    )),
                }
            });
            match compiler.compile_into_spirv(&glsl_code, kind, path.to_str().unwrap(), SHADER_ENTRY_POINT_NAME, Some(&options)) {
                Ok(compile_result) => {
                    if compile_result.get_num_warnings() > 0 {
                        warn!("warnings when compiling {:?}:\n{}", path, compile_result.get_warning_messages());
                    }
                    compile_result
                }
                Err(compile_error) => {
                    error!("{}", compile_error);
                    return Err(());
                }
            }
        };

        Ok(device.create_shader_module(wgpu::ShaderModuleSource::SpirV(compilation_artifact.as_binary())))
    }
}
