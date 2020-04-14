// Workaround for bug in shaderc when using textureSize/texelFetch on textures (instead of samplers)
// error: 'textureSize' : required extension not requested: GL_EXT_samplerless_texture_functions
// Shouldn't happen since we compile for Vulkan.
#extension GL_EXT_samplerless_texture_functions : require

// Occupancy calculator: https://xmartlabs.github.io/cuda-calculator/
#define COMPUTE_PASS_PARTICLES layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;
#define COMPUTE_PASS_VOLUME layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

layout(set = 1, binding = 0) uniform SimulationProperties { uint NumParticles; };
