#version 450

#include "per_frame_resources.glsl"
#include "utilities.glsl"

layout(location = 0) in vec4 in_Color;
layout(location = 0) out vec4 out_Color;

void main() { out_Color = in_Color; }
