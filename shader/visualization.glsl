#include "simulation/particles.glsl"

layout(set = 1, binding = 0) buffer restrict ParticleBuffer { Particle Particles[]; };
layout(set = 1, binding = 1) uniform texture3D VelocityVolume;
layout(set = 1, binding = 2) uniform utexture3D MarkerVolume;
layout(set = 1, binding = 3) uniform texture3D DivergenceVolume;
layout(set = 1, binding = 4) uniform texture3D PressureVolume;

ivec3 getVolumeCoordinate(uint positionIndex) {
    ivec3 volumeSize = textureSize(VelocityVolume, 0).xyz;
    return ivec3(positionIndex % volumeSize.x, positionIndex / volumeSize.x % volumeSize.y, positionIndex / volumeSize.x / volumeSize.y);
}