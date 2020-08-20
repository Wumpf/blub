#include "per_frame_resources.glsl"
#include "simulation/hybrid_fluid.glsl"
#include "simulation/particles.glsl"
#include "utilities.glsl"

// Staggered velocity volume with marker at the center and velocity components on the positive walls.
layout(set = 2, binding = 0, r32f) uniform restrict image3D VelocityVolumeX;
layout(set = 2, binding = 1, r32f) uniform restrict image3D VelocityVolumeY;
layout(set = 2, binding = 2, r32f) uniform restrict image3D VelocityVolumeZ;
layout(set = 2, binding = 3) uniform texture3D PressureVolume;

// Perf note: Tried to encode direct positive neighbars in Marker volume (making it rgba8_snorm) in order to eliminate extra sampling in various
// places. Immediate effect was a lot worse perf. Has texture cache optimizations for R8 to leverage the packed size? Result quite unexpected. Unclear
// if there particular wins/losses.
layout(set = 3, binding = 0) uniform texture3D MarkerVolume;
layout(set = 3, binding = 1, r8_snorm) uniform restrict image3D MarkerVolumeWrite;
