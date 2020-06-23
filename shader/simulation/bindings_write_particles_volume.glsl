#include "per_frame_resources.glsl"
#include "simulation/hybrid_fluid.glsl"
#include "simulation/particles.glsl"

layout(set = 2, binding = 0) buffer restrict ParticlePositionLlBuffer { ParticlePositionLl Particles[]; };
layout(set = 2, binding = 1) buffer restrict ParticleBufferVx { vec4 ParticleBufferVelocityX[]; };
layout(set = 2, binding = 2) buffer restrict ParticleBufferVy { vec4 ParticleBufferVelocityY[]; };
layout(set = 2, binding = 3) buffer restrict ParticleBufferVz { vec4 ParticleBufferVelocityZ[]; };

// Staggered velocity volume with marker at the center and velocity components on the positive walls.
layout(set = 2, binding = 4, r32f) uniform restrict image3D VelocityVolumeX;
layout(set = 2, binding = 5, r32f) uniform restrict image3D VelocityVolumeY;
layout(set = 2, binding = 6, r32f) uniform restrict image3D VelocityVolumeZ;

// Marker grid
layout(set = 2, binding = 7, r8ui) uniform restrict coherent uimage3D MarkerVolume;

// Pressure volume.
layout(set = 2, binding = 8) uniform texture3D PressureVolume;
