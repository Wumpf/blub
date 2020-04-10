// Reading an image out of bounds returns 0, this is why all linked list pointers on the grid are offset by +1
// Otherwise this is the value for an invalid linked list ptr.
#define INVALID_LINKED_LIST_PTR 0xFFFFFFFF

#ifdef NEED_PARTICLE_WRITE
#define PARTICLE_ACCESS restrict
#else
#define PARTICLE_ACCESS restrict readonly
#endif

#ifdef NEED_LLGRID_WRITE
#define LLGRID_ACCESS restrict
#else
#define LLGRID_ACCESS restrict readonly
#endif

struct Particle {
    // Particle positions are in grid space to simplify shader computation
    // (no scaling/translation needed until we're rendering or interacting with other objects!)
    // Grid coordinates are non-fractional.
    vec3 Position;
    uint LinkedListNext;
    vec3 Velocity;
    float Padding1;
};

// todo: would be nice to do readonly when it's bound readonly!
layout(set = 1, binding = 0) buffer PARTICLE_ACCESS ParticleBuffer { Particle Particles[]; };

// Fluid volumes (both variables point to different ones!)
layout(set = 2, binding = 0, rgba32f) uniform restrict image3D VelocityGridWrite;
layout(set = 2, binding = 1) uniform texture3D VelocityGridRead;
// Dual grid linked list
// Our dual grid has a positive 0.5 offset. I.e. its origin relative to the main grid is at (0.5, 0.5, 0.5)
// This also means that the main grid cell at (0,0,0) has only one neighbor in dual, whereas the cell at (gridsize-1) has all 8
layout(set = 3, binding = 0, r32ui) uniform LLGRID_ACCESS uimage3D LinkedListDualGrid;

// TODO: Set local_size, don't leave it at 1 because that might be really bad https://xmartlabs.github.io/cuda-calculator/