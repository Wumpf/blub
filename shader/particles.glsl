// Reading an image out of bounds returns 0, this is why all linked list pointers on the grid are offset by +1
// Otherwise this is the value for an invalid linked list ptr.
#define INVALID_LINKED_LIST_PTR 0xFFFFFFFF

struct Particle {
    // Particle positions are in grid space to simplify shader computation
    // (no scaling/translation needed until we're rendering or interacting with other objects!)
    // Grid coordinates are non-fractional.
    vec3 Position;
    uint LinkedListNext;

    // 3x3 Velocity jacobi matrix (APIC) + velocity vector (column 4)
    // We want a mat4x3 with row major storage. This works with special layout attributes but assignment is still weird, so let's play this safe!
    vec4 VelocityMatrix[3];
};
