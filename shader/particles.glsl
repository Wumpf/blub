// Reading an image out of bounds returns 0, this is why all linked list pointers on the grid are offset by +1
// Otherwise this is the value for an invalid linked list ptr.
#define INVALID_LINKED_LIST_PTR 0xFFFFFFFF

#ifndef PARTICLE_WRITEACCESS
#define PARTICLE_ACCESS restrict readonly
#else
#define PARTICLE_ACCESS restrict PARTICLE_WRITEACCESS
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

layout(set = 1, binding = 0) buffer PARTICLE_ACCESS ParticleBuffer { Particle Particles[]; };

// Occupancy calculator: https://xmartlabs.github.io/cuda-calculator/
#define COMPUTE_PASS_PARTICLES layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;
