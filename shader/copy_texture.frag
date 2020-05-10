#version 450

#include "utilities.glsl"

layout(set = 0, binding = 0) uniform texture2D Texture;

layout(location = 0) out vec4 out_Color;

void main() { out_Color = texelFetch(Texture, ivec2(gl_FragCoord.xy), 0); }