#include "global_bindings.glsl"
#include "simulation/hybrid_fluid.glsl"
#include "simulation/particles.glsl"
#include "utilities.glsl"

// Perf note: Tried to encode direct positive neighbars in Marker volume (making it rgba8_snorm) in order to eliminate extra sampling in various
// places. Immediate effect was a lot worse perf.
layout(set = 2, binding = 0) uniform texture3D MarkerVolume;

// Staggered velocity volume with marker at the center and velocity components on the positive walls.
layout(set = 2, binding = 1, r32f) uniform restrict image3D VelocityVolumeX;
layout(set = 2, binding = 2, r32f) uniform restrict image3D VelocityVolumeY;
layout(set = 2, binding = 3, r32f) uniform restrict image3D VelocityVolumeZ;
layout(set = 2, binding = 4) uniform texture3D PressureVolume;
