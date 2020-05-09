#version 450

#include "per_frame_resources.glsl"

layout(location = 0) in vec3 in_Position;
layout(location = 1) in vec3 in_Color;

out gl_PerVertex { vec4 gl_Position; };

layout(location = 0) out vec3 out_Color;

void main() {
    out_Color = in_Color;
    gl_Position = Camera.ViewProjection * vec4(in_Position, 1.0);
}
