#version 450

// Workaround for bug in shaderc when using textureSize/texelFetch on textures (instead of samplers)
// error: 'texelFetch' : required extension not requested: GL_EXT_samplerless_texture_functions
// Shouldn't happen since we compile for Vulkan.
#extension GL_EXT_samplerless_texture_functions : require

layout(set = 0, binding = 0) uniform texture2D Texture;

layout(location = 0) out vec4 out_Color;

void main() { out_Color = texelFetch(Texture, ivec2(gl_FragCoord.xy), 0); }