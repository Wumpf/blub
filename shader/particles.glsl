struct Particle
{
    vec3 Position;
    float Padding0;
    vec3 Velocity;
    float Padding1;
};

layout(binding = 1) buffer ParticleBuffer
{
    Particle Particles[];
};
