#version 460

#include "global_bindings.glsl"

layout(location = 0) in vec3 in_Position;
layout(push_constant) uniform PushConstants_ { uint MeshIndex; };

// layout(location = 0) out vec3 out_GridPosition;

out gl_PerVertex { vec4 gl_Position; };

void main() {
    // todo
    vec3 worldPosition = (vec4(in_Position, 1.0) * Meshes[MeshIndex].Transform).xyz;
    gl_Position = Camera.ViewProjection * vec4(worldPosition, 1.0);
}
