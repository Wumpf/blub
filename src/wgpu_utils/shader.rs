use notify::Watcher;
use std::{borrow::Cow::Borrowed, cell::RefCell};
use std::{ffi::OsStr, hash::Hash};
use std::{hash::Hasher, sync::Arc};
use std::{
    path::{Path, PathBuf},
    sync::Mutex,
};

// All entry points need to have this name.
// (could make customizable, but forcing this has perks as well)
pub const SHADER_ENTRY_POINT_NAME: &str = "main";

pub struct ShaderDirectory {
    #[allow(dead_code)]
    watcher: notify::RecommendedWatcher,
    changed_files: Arc<Mutex<Vec<PathBuf>>>,
    directory: PathBuf,
    cache_dir: PathBuf,
}

pub struct ShaderModuleWithSourceFiles {
    pub module: wgpu::ShaderModule,
    pub source_files: Vec<PathBuf>, // main source file and all includes
}

impl ShaderDirectory {
    pub fn new(path: &Path, cache_dir: &Path) -> ShaderDirectory {
        let changed_files = Arc::new(Mutex::new(Vec::<PathBuf>::new()));
        let changed_files_evt_ref = changed_files.clone();
        let mut watcher: notify::RecommendedWatcher = notify::Watcher::new_immediate(move |res: Result<notify::Event, notify::Error>| match res {
            Ok(evt) => match evt.kind {
                notify::EventKind::Any => {}
                notify::EventKind::Access(_) => {}
                notify::EventKind::Create(_) => {}
                notify::EventKind::Modify(_) => {
                    let mut changes = changed_files_evt_ref.lock().unwrap();
                    for path in evt.paths.iter() {
                        if !path.is_file() || changes.contains(path) {
                            continue;
                        }
                        changes.push(path.canonicalize().unwrap());
                    }
                }
                notify::EventKind::Remove(_) => {}
                notify::EventKind::Other => {}
            },
            Err(e) => error!("Failed to create filewatcher: {:?}", e),
        })
        .unwrap();
        watcher.watch(path, notify::RecursiveMode::Recursive).unwrap();

        let cache_dir = PathBuf::from(cache_dir).join(if cfg!(debug_assertions) { "debug" } else { "release" });
        let _ = std::fs::create_dir_all(&cache_dir);

        ShaderDirectory {
            watcher,
            changed_files,
            directory: PathBuf::from(path),
            cache_dir,
        }
    }

    // Checks if any change was detected in the shader directory.
    // Right now notifies any changes in the directory, if too slow consider filtering & distinguishing shaders.
    pub fn drain_changed_files(&self) -> Vec<PathBuf> {
        self.changed_files.lock().unwrap().drain(..).collect()
    }

    pub fn load_shader_module(&self, device: &wgpu::Device, relative_path: &Path) -> Result<ShaderModuleWithSourceFiles, ()> {
        let path = self.directory.join(relative_path);
        let source_files = RefCell::new(vec![path.canonicalize().unwrap()]);

        let glsl_code = match std::fs::read_to_string(&path) {
            Ok(glsl_code) => glsl_code,
            Err(err) => {
                error!("Failed to read shader file \"{:?}\": {}", path, err);
                return Err(());
            }
        };

        let kind = match path.extension().and_then(OsStr::to_str) {
            Some("frag") => shaderc::ShaderKind::Fragment,
            Some("vert") => shaderc::ShaderKind::Vertex,
            Some("comp") => shaderc::ShaderKind::Compute,
            _ => {
                error!("Did not recognize file extension for shader file \"{:?}\"", path);
                return Err(());
            }
        };

        // Check for cache hit.
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        glsl_code.hash(&mut hasher);

        let cache_path = self.cache_dir.join(format!(
            "{:X}.{}.cache",
            hasher.finish(),
            path.extension().and_then(OsStr::to_str).unwrap()
        ));
        let dependent_sources_cache_path = cache_path.with_extension("files.cache");
        if let Ok(cached_shader) = std::fs::read(&cache_path) {
            if let Ok(sources_string) = std::fs::read_to_string(&dependent_sources_cache_path) {
                return Ok(ShaderModuleWithSourceFiles {
                    module: device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                        label: Some(path.file_name().unwrap().to_str().unwrap()),
                        source: wgpu::ShaderSource::SpirV(Borrowed(bytemuck::cast_slice(&cached_shader))),
                        flags: wgpu::ShaderFlags::empty(),
                    }),
                    source_files: sources_string.lines().map(|line| PathBuf::from(line)).collect(),
                });
            }
        }

        let compilation_artifact = {
            let mut compiler = shaderc::Compiler::new().unwrap();
            let mut options = shaderc::CompileOptions::new().unwrap();
            //options.set_hlsl_io_mapping(true);
            options.set_warnings_as_errors();
            options.set_target_env(shaderc::TargetEnv::Vulkan, 0);
            //if cfg!(debug_assertions) {
            //    options.set_optimization_level(shaderc::OptimizationLevel::Performance);
            //} else {
            options.set_optimization_level(shaderc::OptimizationLevel::Performance);
            //}
            // Helps a lot when inspecting in ShaderDoc (will show all original source files before processing) but doesn't seem to hurt performance at all :)
            options.set_generate_debug_info();

            options.add_macro_definition("FRAGMENT_SHADER", Some(if kind == shaderc::ShaderKind::Fragment { "1" } else { "0" }));
            options.add_macro_definition("VERTEX_SHADER", Some(if kind == shaderc::ShaderKind::Vertex { "1" } else { "0" }));
            options.add_macro_definition("COMPUTE_SHADER", Some(if kind == shaderc::ShaderKind::Compute { "1" } else { "0" }));

            if cfg!(debug_assertions) {
                options.add_macro_definition("DEBUG", Some("1"));
            } else {
                options.add_macro_definition("NDEBUG", Some("1"));
            }

            options.set_include_callback(|name, include_type, source_file, _depth| {
                let path = if include_type == shaderc::IncludeType::Relative {
                    Path::new(Path::new(source_file).parent().unwrap()).join(name)
                } else {
                    self.directory.join(name)
                };
                match std::fs::read_to_string(&path) {
                    Ok(glsl_code) => {
                        source_files.borrow_mut().push(path.canonicalize().unwrap());
                        Ok(shaderc::ResolvedInclude {
                            resolved_name: String::from(name),
                            content: glsl_code,
                        })
                    }
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
                    error!("failed to compile shader {:?}: {}", path, compile_error);
                    return Err(());
                }
            }
        };

        std::fs::write(&cache_path, compilation_artifact.as_binary_u8()).or_else(|e| {
            error!("failed to shader cache file {:?}: {}", cache_path, e);
            Err(())
        })?;
        std::fs::write(
            &dependent_sources_cache_path,
            source_files
                .borrow()
                .iter()
                .map(|path| path.to_str().unwrap())
                .collect::<Vec<&str>>()
                .join("\n"),
        )
        .or_else(|e| {
            error!("failed to shader cache dependency file {:?}: {}", dependent_sources_cache_path, e);
            Err(())
        })?;

        Ok(ShaderModuleWithSourceFiles {
            module: device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: Some(path.file_name().unwrap().to_str().unwrap()),
                source: wgpu::ShaderSource::SpirV(Borrowed(&compilation_artifact.as_binary())),
                flags: wgpu::ShaderFlags::empty(),
            }),
            source_files: source_files.into_inner(),
        })
    }
}
