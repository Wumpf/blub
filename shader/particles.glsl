// Reading an image out of bounds returns 0, this is why we use the otherwise inconvenient 0 as invalid!
#define INVALID_LINKED_LIST_PTR 0

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
layout(set = 1, binding = 0) buffer ParticleBuffer { Particle Particles[]; };

// Fluid volume
layout(set = 2, binding = 0, rgba32f) uniform restrict image3D VelocityGridWrite;
layout(set = 2, binding = 1) uniform texture3D VelocityGridRead;
// Dual grid linked list
// Our dual grid has a positive 0.5 offset. I.e. its origin relative to the main grid is at (0.5, 0.5, 0.5)
// This also means that the main grid cell at (0,0,0) has only one neighbor in dual, whereas the cell at (gridsize-1) has all 8
layout(set = 3, binding = 0, r32ui) uniform restrict uimage3D LinkedListDualGrid; // access is sometimes readonly, consider modifying conditionally
