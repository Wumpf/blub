#version 460

#include "mesh.glsl"
#include "per_frame_resources.glsl"

layout(location = 0) in vec3 in_Position;
layout(location = 1) in vec3 in_Normal;
layout(location = 2) in vec2 in_Texcoord;

layout(location = 0) out vec3 out_Normal;

out gl_PerVertex { vec4 gl_Position; };

void main() {
    out_Normal = in_Normal;
    gl_Position = Camera.ViewProjection * (Meshes[MeshIndex].Transform * vec4(in_Position, 1.0));
}
