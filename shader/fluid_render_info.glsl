#include "simulation/particles.glsl"
#include "utilities.glsl"

// Since this is quite a few descriptors, this is a good time to remember that amount of descriptors per layout isn't problematic nowadays.
// See:
// * "One Set Design", slide 10 https://gpuopen.com/wp-content/uploads/2016/03/VulkanFastPaths.pdf
// * "For Tier 3 do keep your unused descriptors bound – don’t waste time unbinding them as this can easily introduce state thrashing bottlenecks"
//    https://developer.nvidia.com/dx12-dos-and-donts  (Tier3 starts with Pascal (10xx) series with Nvidia)
// What _is_ expensive is a big descriptor _set_ layout, but wgpu limits us to 4 descriptor sets anyways :)

layout(set = 1, binding = 0) buffer restrict readonly ParticlePositionLlBuffer { ParticlePositionLl Particles[]; };
layout(set = 1, binding = 1) buffer restrict readonly ParticleBufferVx { vec4 ParticleBufferVelocityX[]; };
layout(set = 1, binding = 2) buffer restrict readonly ParticleBufferVy { vec4 ParticleBufferVelocityY[]; };
layout(set = 1, binding = 3) buffer restrict readonly ParticleBufferVz { vec4 ParticleBufferVelocityZ[]; };
layout(set = 1, binding = 4) uniform texture3D VelocityVolumeX;
layout(set = 1, binding = 5) uniform texture3D VelocityVolumeY;
layout(set = 1, binding = 6) uniform texture3D VelocityVolumeZ;
layout(set = 1, binding = 7) uniform texture3D MarkerVolume;
layout(set = 1, binding = 8) uniform texture3D PressureVolume_Velocity;
layout(set = 1, binding = 9) uniform texture3D PressureVolume_Density;
#ifdef DEBUG
layout(set = 1, binding = 10) uniform texture3D DebugVolume;
#endif
