// Occupancy calculator: https://xmartlabs.github.io/cuda-calculator/
#define COMPUTE_PASS_PARTICLES layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;
#define COMPUTE_PASS_VOLUME layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

// todo: gravity
layout(set = 1, binding = 0) uniform SimulationProperties {
    vec3 GravityGridSpace;
    uint NumParticles;
};

// Boundary is zero, so texel fetch outside of the domain always gives us boundary cells.
#define CELL_SOLID 0 // A couple of things rely on this being zero! (sampling images out of bounds returns zero)
#define CELL_FLUID 1
#define CELL_AIR 2

// TODO: Idea: Cell marker could encode direct neighborhood and thus safe us quite a few samplings!
// TODO: Idea: Put cell marker into w channel of velocity volume redundantly