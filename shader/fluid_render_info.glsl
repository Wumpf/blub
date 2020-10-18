#include "simulation/particles.glsl"
#include "utilities.glsl"

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

ivec3 getVolumeCoordinate(uint positionIndex) {
    ivec3 volumeSize = textureSize(PressureVolume_Velocity, 0).xyz;
    return ivec3(positionIndex % volumeSize.x, positionIndex / volumeSize.x % volumeSize.y, positionIndex / volumeSize.x / volumeSize.y);
}