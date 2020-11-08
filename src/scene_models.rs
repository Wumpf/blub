use cgmath::EuclideanSpace;
use serde::Deserialize;
use std::{error::Error, path::Path, path::PathBuf};
use wgpu::util::DeviceExt;

// Data describing a model in the scene.
#[derive(Deserialize)]
pub struct StaticObjectConfig {
    pub model: PathBuf,
    pub world_position: cgmath::Point3<f32>,
    pub scale: f32,
    pub rotation_angles: cgmath::Point3<cgmath::Deg<f32>>,
}

pub struct MeshData {
    pub transform: cgmath::Matrix4<f32>,            // todo? 3x4 is enough (rotation, scale, translation)
    pub vertex_buffer_range: core::ops::Range<u32>, // range in number of vertices (not bytes!)
    pub index_buffer_range: core::ops::Range<u32>,  // range in number of indices (not bytes!)

                                                    // Material data. If we expected many materials would share a transform this would be a bad idea to put it together.
                                                    // But per loaded mesh we typically only have one.
                                                    // todo: add things like
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MeshDataGpu {
    transform: cgmath::Matrix4<f32>, // todo: Make this a 4x3
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

// Data for _all_ meshes/models in a scene.
pub struct SceneModels {
    // Since we don't add/remove models while running, we can put everything into a single large vertex+index buffer
    pub index_buffer: wgpu::Buffer,
    pub vertex_buffer: wgpu::Buffer,
    pub mesh_desc_buffer: wgpu::Buffer,

    pub meshes: Vec<MeshData>,
}

impl SceneModels {
    pub fn from_config(device: &wgpu::Device, configs: &Vec<StaticObjectConfig>) -> Result<Self, Box<dyn Error>> {
        let mut vertices = Vec::new();
        let mut indices = Vec::<u32>::new();
        let mut meshes = Vec::new();

        for static_object_config in configs {
            let file_name = Path::new("models").join(&static_object_config.model);
            let (mut loaded_models, _loaded_materials) = tobj::load_obj(&file_name, true)?;

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
                    meshes.push(MeshData {
                        transform: cgmath::Matrix4::from_translation(static_object_config.world_position.to_vec())
                            * cgmath::Matrix4::from_scale(static_object_config.scale)
                            * cgmath::Matrix4::from_angle_x(static_object_config.rotation_angles.x)
                            * cgmath::Matrix4::from_angle_y(static_object_config.rotation_angles.y)
                            * cgmath::Matrix4::from_angle_z(static_object_config.rotation_angles.z),
                        vertex_buffer_range: (vertices.len() as u32)..(vertices.len() as u32),
                        index_buffer_range: (indices.len() as u32)..(indices.len() as u32),
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
                    vertex.uv.y = uv[1];
                }
            }
        }

        let meshes_gpu: Vec<MeshDataGpu> = meshes.iter().map(|mesh| MeshDataGpu { transform: mesh.transform }).collect();

        Ok(SceneModels {
            vertex_buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SceneModel VertexBuffer"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsage::VERTEX,
            }),
            index_buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SceneModel IndexBuffer"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsage::INDEX,
            }),
            mesh_desc_buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SceneModel Mesh Data"),
                contents: bytemuck::cast_slice(&meshes_gpu),
                usage: wgpu::BufferUsage::STORAGE,
            }),
            meshes,
        })
    }
}
