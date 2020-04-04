struct Particle {
    vec3 Position;
    float Padding0;
    vec3 Velocity;
    float Padding1;
};

// todo: would be nice to do readonly when it's bound readonly!
layout(set = 0, binding = 0) buffer ParticleBuffer { Particle Particles[]; };

// Fluid volume
layout(set = 1, binding = 0, rgba32f) uniform restrict image3D VelocityGridWrite;
layout(set = 1, binding = 1) uniform texture3D VelocityGridRead;
layout(set = 1, binding = 2) uniform sampler SamplerTrilinear;
