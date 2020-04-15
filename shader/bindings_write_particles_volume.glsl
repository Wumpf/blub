#include "hybrid_fluid.glsl"
#include "particles.glsl"
#include "per_frame_resources.glsl"

layout(set = 2, binding = 0) buffer restrict ParticleBuffer { Particle Particles[]; };

// Origin texel represents position (0,0,0), NOT as one might expect (-0.5 / textureSize)
layout(set = 2, binding = 1, rgba32f) uniform restrict image3D VelocityVolume;

// Dual grid linked list
// Our dual grid has a positive 0.5 offset. I.e. its origin relative to the main grid is at (0.5, 0.5, 0.5)
// This also means that the main grid cell at (0,0,0) has only one neighbor in dual, whereas the cell at (gridsize-1) has all 8
layout(set = 2, binding = 2, r32ui) uniform restrict coherent uimage3D LinkedListDualGrid;

layout(set = 2, binding = 3) uniform texture3D PressureVolume;

// Reading an image out of bounds returns 0, this is why all linked list pointers on the grid are offset by +1
// Otherwise this is the value for an invalid linked list ptr.
#define INVALID_LINKED_LIST_PTR 0xFFFFFFFF