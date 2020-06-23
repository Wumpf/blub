// Reading an image out of bounds returns 0, this is why all linked list pointers on the grid are offset by +1
// Otherwise this is the value for an invalid linked list ptr.
#define INVALID_LINKED_LIST_PTR 0xFFFFFFFF

struct ParticlePositionLl {
    // Particle positions are in grid space to simplify shader computation
    // (no scaling/translation needed until we're rendering or interacting with other objects!)
    // Grid coordinates are non-fractional.
    vec3 Position;
    uint LinkedListNext;
};

// Every particle also has 3x float4 to store the affine velocity matrix (APIC!)
// Experiments have shown that this split up is considerably faster for transfer_build_linkedlist and update_particles (and slightly slower for
// transfer_gather).
// (Speedup of transfer_build_linkedlist makes a lot of sense but speedup of update_particles is unclear!)