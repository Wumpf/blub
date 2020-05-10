// Adds [[flatten]] and friends (which are understood by SPIR-V)
// https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GL_EXT_control_flow_attributes.txt
#extension GL_EXT_control_flow_attributes : require

// Workaround for bug in shaderc when using textureSize/texelFetch on textures (instead of samplers)
// error: 'textureSize' : required extension not requested: GL_EXT_samplerless_texture_functions
// Shouldn't happen since we compile for Vulkan.
#extension GL_EXT_samplerless_texture_functions : require

float saturate(float x) { return clamp(x, 0.0, 1.0); }
vec2 saturate(vec2 x) { return clamp(x, vec2(0.0), vec2(1.0)); }
vec3 saturate(vec3 x) { return clamp(x, vec3(0.0), vec3(1.0)); }
vec4 saturate(vec4 x) { return clamp(x, vec4(0.0), vec4(1.0)); }

float lengthsq(vec3 a, vec3 b) {
    vec3 v = a - b;
    return dot(v, v);
}

float sq(float a) { return a * a; }
vec2 sq(vec2 a) { return a * a; }
vec3 sq(vec3 a) { return a * a; }
vec4 sq(vec4 a) { return a * a; }