#include "per_frame_resources.glsl"
#include "simulation/hybrid_fluid.glsl"
#include "simulation/particles.glsl"

layout(set = 2, binding = 0) buffer restrict ParticlePositionLlBuffer { ParticlePositionLl Particles[]; };
layout(set = 2, binding = 1) buffer restrict readonly ParticleComp { vec4 ParticleBufferVelocityComponent[]; };
layout(set = 2, binding = 2, r32ui) uniform restrict uimage3D LinkedListDualGrid;
layout(set = 2, binding = 3, r8_snorm) uniform restrict image3D MarkerVolume;
layout(set = 2, binding = 4, r32f) uniform restrict image3D VelocityComponentVolume;
layout(push_constant) uniform PushConstants { uint VelocityTransferComponent; };

// Reading an image out of bounds returns 0, this is why all linked list pointers on the grid are offset by +1
// Otherwise this is the value for an invalid linked list ptr.
#define INVALID_LINKED_LIST_PTR 0xFFFFFFFF