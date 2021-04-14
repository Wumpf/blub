#version 460

#include "global_bindings.glsl"

layout(location = 0) in vec3 in_Position;
layout(location = 1) in vec3 in_Normal;
layout(location = 2) in vec2 in_Texcoord;
layout(push_constant) uniform PushConstants_ { uint MeshIndex; };

layout(location = 0) out vec3 out_Normal;
layout(location = 1) out vec2 out_Texcoord;

out gl_PerVertex { vec4 gl_Position; };

void main() {
    out_Normal = in_Normal;
    out_Texcoord = in_Texcoord;
    gl_Position = Camera.ViewProjection * vec4(vec4(in_Position, 1.0) * Meshes[MeshIndex].WorldTransform, 1.0);
}
