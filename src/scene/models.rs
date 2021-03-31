use cgmath::{EuclideanSpace, Matrix, SquareMatrix};
use serde::Deserialize;
use std::{error::Error, path::Path, path::PathBuf, time::Duration};
use wgpu::util::DeviceExt;

use crate::timer::Timer;

// Data describing a model in the scene.
#[derive(Deserialize, Clone)]
pub struct StaticObjectConfig {
    pub model: PathBuf,
    pub world_position: cgmath::Point3<f32>,
    pub scale: f32,
    pub rotation_angles: cgmath::Point3<cgmath::Deg<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub animation: Option<RigidAnimation>,
}

#[derive(Deserialize, Clone)]
pub enum AnimationCurve {
    Linear,
}

#[derive(Deserialize, Clone)]
pub struct RigidAnimation {
    pub translation_target: cgmath::Point3<f32>,
    pub translation_curve: AnimationCurve,
    pub translation_duration: f32, // time to reach the target_position in seconds
                                   //pub rotational_speed: cgmath::Vector3<cgmath::Deg<f32>>, // TODO
}

pub struct StaticMeshData {
    pub config: StaticObjectConfig,

    pub vertex_buffer_range: core::ops::Range<u32>, // range in number of vertices (not bytes!)
    pub index_buffer_range: core::ops::Range<u32>,  // range in number of indices (not bytes!)

    // Material data. If we expected many materials would share a transform this would be a bad idea to put it together.
    // But per loaded mesh we typically only have one.
    pub texture_index: i32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct MeshDataGpu {
    transform_r0: cgmath::Vector4<f32>,
    transform_r1: cgmath::Vector4<f32>,
    transform_r2: cgmath::Vector4<f32>,
    step_translation: cgmath::Vector3<f32>,
    padding0: f32,

    vertex_buffer_range: cgmath::Vector2<u32>,
    index_buffer_range: cgmath::Vector2<u32>,

    texture_index: i32,
    padding1: cgmath::Vector3<i32>,
}
unsafe impl bytemuck::Pod for MeshDataGpu {}
unsafe impl bytemuck::Zeroable for MeshDataGpu {}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MeshVertex {
    position: cgmath::Point3<f32>,
    normal: cgmath::Vector3<f32>,
    uv: cgmath::Point2<f32>,
}
unsafe impl bytemuck::Pod for MeshVertex {}
unsafe impl bytemuck::Zeroable for MeshVertex {}
impl Default for MeshVertex {
    fn default() -> Self {
        MeshVertex {
            position: cgmath::point3(0.0, 0.0, 0.0),
            normal: cgmath::Zero::zero(),
            uv: cgmath::point2(0.0, 0.0),
        }
    }
}

impl MeshVertex {
    pub const SIZE: wgpu::BufferAddress = std::mem::size_of::<MeshVertex>() as wgpu::BufferAddress;
}

// Data for _all_ meshes/models in a scene.
pub struct SceneModels {
    // Since we don't add/remove models while running, we can put everything into a single large vertex+index buffer
    pub index_buffer: wgpu::Buffer,
    pub vertex_buffer: wgpu::Buffer,
    pub mesh_desc_buffer: wgpu::Buffer,

    pub texture_views: Vec<wgpu::TextureView>,

    pub meshes: Vec<StaticMeshData>,
}

fn load_texture2d_from_path(device: &wgpu::Device, queue: &wgpu::Queue, path: &Path) -> wgpu::Texture {
    info!("Loading 2d texture {:?}", path);
    // TODO: Mipmaps

    let image = image::io::Reader::open(path).unwrap().decode().unwrap().to_rgba8();
    let image_data = image.as_raw();

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: path.file_name().unwrap().to_str(),
        size: wgpu::Extent3d {
            width: image.width(),
            height: image.height(),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
    });

    queue.write_texture(
        wgpu::TextureCopyView {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
        },
        &image_data,
        wgpu::TextureDataLayout {
            offset: 0,
            bytes_per_row: 4 * image.width(),
            rows_per_image: image.height(),
        },
        wgpu::Extent3d {
            width: image.width(),
            height: image.height(),
            depth_or_array_layers: 1,
        },
    );

    texture
}

impl StaticMeshData {
    fn world_position_at_time(&self, total_simulated_time: Duration) -> cgmath::Point3<f32> {
        if let Some(animation) = &self.config.animation {
            let mut translation_progress = total_simulated_time.as_secs_f32() % (animation.translation_duration * 2.0);
            if translation_progress > animation.translation_duration {
                translation_progress = animation.translation_duration * 2.0 - translation_progress;
            }
            translation_progress /= animation.translation_duration;

            self.config.world_position * (1.0 - translation_progress) + animation.translation_target.to_vec() * translation_progress
        } else {
            self.config.world_position
        }
    }

    fn to_gpu(&self, total_simulated_time: Duration, simulation_delta: Duration) -> MeshDataGpu {
        let world_position = self.world_position_at_time(total_simulated_time);

        // Brute force way for getting a translation vector. Analytical derivative would probably be better.
        let step_translation = if total_simulated_time > simulation_delta {
            world_position - self.world_position_at_time(total_simulated_time - simulation_delta)
        } else {
            cgmath::vec3(0.0, 0.0, 0.0)
        };

        let transform = cgmath::Matrix4::from_translation(world_position.to_vec())
            * cgmath::Matrix4::from_scale(self.config.scale)
            * cgmath::Matrix4::from_angle_x(self.config.rotation_angles.x)
            * cgmath::Matrix4::from_angle_y(self.config.rotation_angles.y)
            * cgmath::Matrix4::from_angle_z(self.config.rotation_angles.z);

        let transposed_transform = transform.transpose();
        // let inverse_transform = transposed_transform
        //     .invert()
        //     .expect("Mesh matrix not invertible, this should never happen");
        MeshDataGpu {
            transform_r0: transposed_transform.x,
            transform_r1: transposed_transform.y,
            transform_r2: transposed_transform.z,
            step_translation,
            padding0: 0.0,
            vertex_buffer_range: cgmath::vec2(self.vertex_buffer_range.start, self.vertex_buffer_range.end),
            index_buffer_range: cgmath::vec2(self.index_buffer_range.start, self.index_buffer_range.end),
            texture_index: self.texture_index,
            padding1: cgmath::vec3(0, 0, 0),
        }
    }
}

impl SceneModels {
    pub fn vertex_buffer_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: MeshVertex::SIZE,
            step_mode: wgpu::InputStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 4 * 3,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 4 * 6,
                    shader_location: 2,
                },
            ],
        }
    }

    pub fn from_config(device: &wgpu::Device, queue: &wgpu::Queue, configs: &Vec<StaticObjectConfig>) -> Result<Self, Box<dyn Error>> {
        let mut vertices = Vec::new();
        let mut indices = Vec::<u32>::new();
        let mut meshes = Vec::new();
        let mut texture_paths = Vec::new();

        for static_object_config in configs {
            let file_name = Path::new("models").join(&static_object_config.model);
            let (mut loaded_models, loaded_materials) = tobj::load_obj(&file_name, true)?;

            loaded_models.sort_by_key(|m| m.mesh.material_id);
            let mut prev_material_id = std::usize::MAX;

            // if any mesh in the obj doesn't have a material, we need to add an artificial one and offset all others.
            let missing_materials = loaded_models.iter().any(|m| m.mesh.material_id.is_none());

            for m in loaded_models.iter() {
                let material_id = if missing_materials {
                    match m.mesh.material_id {
                        Some(id) => id + 1,
                        None => 0,
                    }
                } else {
                    m.mesh.material_id.unwrap()
                };
                if prev_material_id != material_id {
                    let texture_index: i32 = if let Some(matid) = m.mesh.material_id {
                        let texture_path = file_name.parent().unwrap().join(&loaded_materials[matid].diffuse_texture);

                        let known_texture_index = texture_paths.iter().position(|p| *p == texture_path);
                        match known_texture_index {
                            Some(index) => index as i32,
                            None => {
                                texture_paths.push(texture_path);
                                texture_paths.len() as i32 - 1
                            }
                        }
                    } else {
                        -1
                    };

                    meshes.push(StaticMeshData {
                        config: static_object_config.clone(),
                        vertex_buffer_range: (vertices.len() as u32)..(vertices.len() as u32),
                        index_buffer_range: (indices.len() as u32)..(indices.len() as u32),
                        texture_index,
                    });
                }
                prev_material_id = material_id;

                indices.extend(&m.mesh.indices);
                let mesh = meshes.last_mut().unwrap();
                mesh.index_buffer_range = mesh.index_buffer_range.start..(indices.len() as u32);

                let prev_vertex_count = vertices.len();
                vertices.resize_with(vertices.len() + m.mesh.positions.len(), || MeshVertex::default());
                mesh.vertex_buffer_range = mesh.vertex_buffer_range.start..(vertices.len() as u32);

                for (vertex, pos) in vertices.iter_mut().skip(prev_vertex_count).zip(m.mesh.positions.chunks(3)) {
                    vertex.position.x = pos[0];
                    vertex.position.y = pos[1];
                    vertex.position.z = pos[2];
                }
                for (vertex, norm) in vertices.iter_mut().skip(prev_vertex_count).zip(m.mesh.normals.chunks(3)) {
                    vertex.normal.x = norm[0];
                    vertex.normal.y = norm[1];
                    vertex.normal.z = norm[2];
                }
                for (vertex, uv) in vertices.iter_mut().skip(prev_vertex_count).zip(m.mesh.texcoords.chunks(2)) {
                    vertex.uv.x = uv[0];
                    vertex.uv.y = 1.0 - uv[1];
                }
            }
        }

        let meshes_gpu: Vec<MeshDataGpu> = meshes
            .iter()
            .map(|mesh| mesh.to_gpu(Duration::from_secs(0), Duration::from_secs(0)))
            .collect();

        let texture_views = texture_paths
            .iter()
            .map(|path| load_texture2d_from_path(device, queue, path).create_view(&Default::default()))
            .collect();

        Ok(SceneModels {
            vertex_buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SceneModel VertexBuffer"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsage::VERTEX | wgpu::BufferUsage::STORAGE,
            }),
            index_buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SceneModel IndexBuffer"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsage::INDEX | wgpu::BufferUsage::STORAGE,
            }),
            mesh_desc_buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SceneModel Mesh Data"),
                contents: bytemuck::cast_slice(&meshes_gpu),
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
            }),
            meshes,
            texture_views,
        })
    }

    pub fn step(&self, timer: &Timer, queue: &wgpu::Queue) {
        // We typically don't have a lot of objects. So just overwrite the entire mesh desc.
        let meshes_gpu: Vec<MeshDataGpu> = self
            .meshes
            .iter()
            .map(|mesh| mesh.to_gpu(timer.total_simulated_time(), timer.simulation_delta()))
            .collect();
        queue.write_buffer(&self.mesh_desc_buffer, 0, bytemuck::cast_slice(&meshes_gpu));
    }
}
