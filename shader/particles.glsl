struct Particle {
    vec3 Position;
    float Padding0;
    vec3 Velocity;
    float Padding1;
};

// todo: would be nice to do readonly when it's bound readonly!
layout(set = 0, binding = 1) buffer ParticleBuffer { Particle Particles[]; };
