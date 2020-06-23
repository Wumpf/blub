#include "per_frame_resources.glsl"
#include "simulation/hybrid_fluid.glsl"
#include "simulation/particles.glsl"
#include "utilities.glsl"

// Staggered velocity volume with marker at the center and velocity components on the positive walls.
layout(set = 2, binding = 0, r32f) uniform restrict image3D VelocityVolumeX;
layout(set = 2, binding = 1, r32f) uniform restrict image3D VelocityVolumeY;
layout(set = 2, binding = 2, r32f) uniform restrict image3D VelocityVolumeZ;

layout(set = 2, binding = 3) uniform utexture3D MarkerVolume;
layout(set = 2, binding = 4) uniform texture3D PressureVolume;
