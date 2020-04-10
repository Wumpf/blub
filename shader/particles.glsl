struct Particle {
    // Particle positions are in grid space to simplify shader computation
    // (no scaling/translation needed until we're rendering or interacting with other objects!)
    // Grid coordinates are non-fractional.
    vec3 Position;
    float Padding0;
    vec3 Velocity;
    float Padding1;
};

// todo: would be nice to do readonly when it's bound readonly!
layout(set = 1, binding = 0) buffer ParticleBuffer { Particle Particles[]; };

// Fluid volume
layout(set = 2, binding = 0, rgba32f) uniform restrict image3D VelocityGridWrite;
layout(set = 2, binding = 1) uniform texture3D VelocityGridRead;

// layout(set = 3, binding = 2, r32ui) uniform restrict uimage3D LinkedListDualGrid;
