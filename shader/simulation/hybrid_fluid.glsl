// Occupancy calculator: https://xmartlabs.github.io/cuda-calculator/
#define COMPUTE_PASS_PARTICLES layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
#define COMPUTE_PASS_VOLUME layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

#ifndef NO_SIMPROPS

layout(set = 1, binding = 0) uniform SimulationProperties {
    vec3 GravityGridSpace;
    uint NumParticles;
};

layout(set = 1, binding = 1) uniform texture3D SceneVoxelization;

#ifdef DEBUG
layout(set = 1, binding = 2, r32f) uniform restrict image3D DebugVolume;
#endif

#endif

// Boundary is zero, so texel fetch outside of the domain always gives us boundary cells.
#define CELL_SOLID 0.0 // A couple of things rely on this being zero! (sampling images out of bounds returns zero)
#define CELL_FLUID 1.0
#define CELL_AIR -1.0

vec3 unpackPushDisplacement(uint packedPushDisplacement) { return unpackSnorm4x8(packedPushDisplacement).xyz * 0.5; }

// A value direct proportional to length of the displacement vector
float displacementLengthComparisionValue(uint packedPushDisplacement) {
    vec3 v = unpackSnorm4x8(packedPushDisplacement).xyz;
    return dot(v, v);
}

uint packPushDisplacement(vec3 displacement) {
    displacement = clamp(displacement, vec3(-0.5), vec3(0.5)); // More than 0.5 cells displacement is not allowed per step.
    return packSnorm4x8(vec4(displacement * 2.0, 0.0));
}