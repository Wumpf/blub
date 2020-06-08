#include "per_frame_resources.glsl"
#include "simulation/hybrid_fluid.glsl"
#include "simulation/particles.glsl"

layout(set = 2, binding = 0) buffer restrict ParticleBuffer { Particle Particles[]; };
// Staggered velocity volume with marker at the center and velocity components on the positive walls.
// Texel at imageCoord(0,0,0) represents cell at grid coord (0.5, 0.5, 0.5)
//
// Note that reading 3 R32f velocity volumes is generally slower than a single RGBA32f. On my Nvidia card 0.5x the speed or R32f
// Other gpus it's even sample RGBA32f as fast as R32f! See https://github.com/sebbbi/perftest#nvidia-maxwell-gtx-980-ti
// But the way our algorithms work, it's much easier to work with three separate volumes.
layout(set = 2, binding = 1, r32f) uniform restrict image3D VelocityVolumeX;
layout(set = 2, binding = 2, r32f) uniform restrict image3D VelocityVolumeY;
layout(set = 2, binding = 3, r32f) uniform restrict image3D VelocityVolumeZ;

// Linked list cells.
// Cell at image coord (0,0,0) shall point to all particles contained in the cell (0,0,0)-(1,1,1)
layout(set = 2, binding = 4, r32ui) uniform restrict coherent uimage3D LinkedListDualGrid;

// Marker grid
layout(set = 2, binding = 5, r8ui) uniform restrict coherent uimage3D MarkerVolume;

// Pressure volume.
layout(set = 2, binding = 6) uniform texture3D PressureVolume;

// Reading an image out of bounds returns 0, this is why all linked list pointers on the grid are offset by +1
// Otherwise this is the value for an invalid linked list ptr.
#define INVALID_LINKED_LIST_PTR 0xFFFFFFFF