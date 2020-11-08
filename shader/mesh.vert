#version 460

#include "per_frame_resources.glsl"

layout(location = 0) in vec3 in_Position;
layout(location = 1) in vec3 in_Normal;
layout(location = 2) in vec2 in_Texcoord;

struct MeshData {
    mat4 Transform;
};
layout(set = 1, binding = 0) restrict readonly buffer Meshes_ { MeshData Meshes[]; };
layout(push_constant) uniform PushConstants_ { uint MeshIndex; };

out gl_PerVertex { vec4 gl_Position; };

void main() { gl_Position = Camera.ViewProjection * (Meshes[MeshIndex].Transform * vec4(in_Position, 1.0)); }
