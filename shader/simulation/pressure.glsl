#define NO_SIMPROPS

#include "per_frame_resources.glsl"
#include "simulation/hybrid_fluid.glsl"
#include "utilities.glsl"

layout(set = 1, binding = 0) uniform texture3D VelocityVolumeX;
layout(set = 1, binding = 1) uniform texture3D VelocityVolumeY;
layout(set = 1, binding = 2) uniform texture3D VelocityVolumeZ;
layout(set = 1, binding = 3) uniform utexture3D MarkerVolume;

struct PcgScalars {
    float AlphaBeta; // after every product this is set to (sigma / dotProductResult)
    float Sigma;     // the dot product between auxilary (preconditioned residual) and residual
    vec2 _Dummy;
};
