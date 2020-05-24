// Occupancy calculator: https://xmartlabs.github.io/cuda-calculator/
#define COMPUTE_PASS_PARTICLES layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;
#define COMPUTE_PASS_VOLUME layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

// todo: gravity
layout(set = 1, binding = 0) uniform SimulationProperties { uint NumParticles; };

// Boundary is zero, so texel fetch outside of the domain always gives us boundary cells.
#define CELL_SOLID 0.0
#define CELL_FLUID 1.0
#define CELL_AIR 2.0

vec3 enforceBoundaryCondition(vec3 velocity, bool isBoundaryCell, bvec3 isBoundaryCellPositiveNeighbor) {

    // Here's something not quite intuitive:
    // If the current cell is marked CELL_SOLID, we still need to preserve velocity between neighboring solid cells!
    // Why?
    // Imagine a vertical wall, any particle falling down directly next to it trilinearly interpolates "into" the solid cell.
    // It's not inside, but in the transfer-to-volume step the vertical velocity was correctly picked up into the wall.
    // If we would meddle with this vertical velocity, the particle will get stuck!
    // TODO: In corners this idea seems to break down. Extrapolate instead?

    // At boundary cells the pressure should be such that fluid_velocity * boundary_normal == boundary_velocity * boundary_normal
    // No support for non-static boundary yet
    // -> No flow between boundary and non-boundary allowed.
    return mix(velocity, vec3(0), notEqual(bvec3(isBoundaryCell), isBoundaryCellPositiveNeighbor));
}

// Once we support arbitrary boundaries, this will likely go away.
vec3 enforceGlobalWallBoundaryCondition(vec3 velocity, ivec3 cellPos, ivec3 gridSize) {
    bvec3 isBoundaryCellPositiveNeighbor = greaterThanEqual(cellPos + ivec3(1), gridSize);
    bool isBoundaryCell = false;
    if (cellPos.x <= 0) {
        isBoundaryCell = true;
        isBoundaryCellPositiveNeighbor.y = true;
        isBoundaryCellPositiveNeighbor.z = true;
    }
    if (cellPos.y <= 0) {
        isBoundaryCell = true;
        isBoundaryCellPositiveNeighbor.x = true;
        isBoundaryCellPositiveNeighbor.z = true;
    }
    if (cellPos.z <= 0) {
        isBoundaryCell = true;
        isBoundaryCellPositiveNeighbor.x = true;
        isBoundaryCellPositiveNeighbor.y = true;
    }

    return enforceBoundaryCondition(velocity, isBoundaryCell, isBoundaryCellPositiveNeighbor);
}
