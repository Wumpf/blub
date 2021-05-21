#include "simulation/particles.glsl"

layout(set = 2, binding = 0) buffer restrict Old_ParticlePositionLlBuffer { ParticlePositionLl Old_Particles[]; };
layout(set = 2, binding = 1) buffer restrict New_ParticlePositionLlBuffer { ParticlePositionLl New_Particles[]; };
layout(set = 2, binding = 2, r32ui) uniform restrict uimage3D ParticleBinningVolume;
layout(set = 2, binding = 3) buffer restrict ParticleBinningAtomicCounter_ { uint ParticleBinningAtomicCounter; };
