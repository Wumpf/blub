#include "per_frame_resources.glsl"
#include "simulation/hybrid_fluid.glsl"
#include "simulation/particles.glsl"

layout(set = 2, binding = 0) buffer restrict ParticleBuffer { Particle Particles[]; };
// Staggered velocity volume with marker at the center and velocity components on the positive walls.
layout(set = 2, binding = 1, r32f) uniform restrict image3D VelocityVolumeX;
layout(set = 2, binding = 2, r32f) uniform restrict image3D VelocityVolumeY;
layout(set = 2, binding = 3, r32f) uniform restrict image3D VelocityVolumeZ;

// Marker grid
layout(set = 2, binding = 4, r8ui) uniform restrict coherent uimage3D MarkerVolume;

// Pressure volume.
layout(set = 2, binding = 5) uniform texture3D PressureVolume;
