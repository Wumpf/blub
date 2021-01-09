#define NO_SIMPROPS

#include "simulation/hybrid_fluid.glsl"
#include "utilities.glsl"

#define COMPUTE_PASS_PRESSURE layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// Properties fo the reduce pass.
#define LOCAL_SIZE_REDUCE 1024
// 32 was distinctively slower, 16 about same as than 8, 4 clearly slower (gtx1070 ti)
#define REDUCE_READS_PER_THREAD 16

layout(set = 0, binding = 0) uniform texture3D MarkerVolume;
layout(set = 1, binding = 0, r32f) uniform restrict image3D Pressure;
layout(set = 1, binding = 1) uniform Config {
    float ErrorTolerance;
    uint MaxNumSolverIterations;
};

struct PcgScalars {
    float AlphaBeta; // after every product this is set to (sigma / dotProductResult)
    float Sigma;     // the dot product between auxiliary (preconditioned residual) and residual
    vec2 _Dummy;
};

layout(push_constant) uniform PushConstants_ {
    uint Mode;             // Used to make adjustments to the shader invocation (which don't justify another shader instance)
    uint SourceBufferSize; // The size of the source buffer
}
PushConstants;

// Result of multiplication with coefficient matrix with a texture at gridCoord.
// Only call if gridCoord is a fluid position!
float MultiplyWithCoefficientMatrix(ivec3 gridCoord, texture3D texture, float valueAtGridCoord) {
    float result = 0.0;
    float markerX0 = texelFetch(MarkerVolume, gridCoord - ivec3(1, 0, 0), 0).x;
    float markerX1 = texelFetch(MarkerVolume, gridCoord + ivec3(1, 0, 0), 0).x;
    float markerY0 = texelFetch(MarkerVolume, gridCoord - ivec3(0, 1, 0), 0).x;
    float markerY1 = texelFetch(MarkerVolume, gridCoord + ivec3(0, 1, 0), 0).x;
    float markerZ0 = texelFetch(MarkerVolume, gridCoord - ivec3(0, 0, 1), 0).x;
    float markerZ1 = texelFetch(MarkerVolume, gridCoord + ivec3(0, 0, 1), 0).x;

    // This is the diagonal value of matrix A!
    float numNonSolidNeighbors = 0.0;
    numNonSolidNeighbors += abs(markerX0); // float(markerX0 != CELL_SOLID);
    numNonSolidNeighbors += abs(markerX1); // float(markerX1 != CELL_SOLID);
    numNonSolidNeighbors += abs(markerY0); // float(markerY0 != CELL_SOLID);
    numNonSolidNeighbors += abs(markerY1); // float(markerY1 != CELL_SOLID);
    numNonSolidNeighbors += abs(markerZ0); // float(markerZ0 != CELL_SOLID);
    numNonSolidNeighbors += abs(markerZ1); // float(markerZ1 != CELL_SOLID);

    // apply diagonal of A
    result += numNonSolidNeighbors * valueAtGridCoord;

    // apply other coefficients of A
    if (markerX0 == CELL_FLUID) {
        result -= texelFetch(texture, gridCoord - ivec3(1, 0, 0), 0).x;
    }
    if (markerX1 == CELL_FLUID) {
        result -= texelFetch(texture, gridCoord + ivec3(1, 0, 0), 0).x;
    }
    if (markerY0 == CELL_FLUID) {
        result -= texelFetch(texture, gridCoord - ivec3(0, 1, 0), 0).x;
    }
    if (markerY1 == CELL_FLUID) {
        result -= texelFetch(texture, gridCoord + ivec3(0, 1, 0), 0).x;
    }
    if (markerZ0 == CELL_FLUID) {
        result -= texelFetch(texture, gridCoord - ivec3(0, 0, 1), 0).x;
    }
    if (markerZ1 == CELL_FLUID) {
        result -= texelFetch(texture, gridCoord + ivec3(0, 0, 1), 0).x;
    }
    return result;
}
