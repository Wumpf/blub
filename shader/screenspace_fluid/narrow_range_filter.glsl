// An implementation of:
// A Narrow-Range Filter for Screen-Space Fluid Rendering, Truong et al. 2018
// http://www.cemyuksel.com/research/papers/narrowrangefilter.pdf

// With some flavoring of my own.

#if !defined(FILTER_1D) && !defined(FILTER_2D)
#error "Need to define either FILTER_1D or FILTER_2D"
#endif

#include "fluid_render_info.glsl"
#include "per_frame_resources.glsl"
#include "utilities.glsl"

layout(set = 2, binding = 0, r32f) uniform restrict image2D DepthDest;
layout(set = 2, binding = 1) uniform texture2D DepthSource;
layout(push_constant) uniform PushConstants { uint FilterDirection; };

// MAX_FILTER_SIZE: Maximum filter size (in one direction, i.e. 15 means that it covers a 31x31 square of pixels when FILTER_XY is active)
#if defined(FILTER_2D)
#define MAX_FILTER_SIZE 5
layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;
#else
#define MAX_FILTER_SIZE 31
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
#endif

// Relationship between the standard deviation sigma (for gaussian kernel) and the filter size.
// (the lower, the boxier the filter gets)
#define SIGMA_TO_FILTERSIZE 3.0

// TODO: Expose
float worldSpaceSigma = 1.5 * Rendering.FluidParticleRadius;
float depthThreshold = 10.0 * Rendering.FluidParticleRadius;
float mu = 10.0 * Rendering.FluidParticleRadius;

// depthSampleA & depthSampleB are depth samples on opposing sides, both at the same distance to the middle.
void narrowRangeFilter(float depthSampleA, float depthSampleB, float lowerDepthBound, float gaussianWeight, float depthThreshold,
                       inout float depthThresholdHigh, inout float depthThresholdLow, inout float totalWeight, inout float filteredDepth) {
    // Is depth too high? Keep filter symmetric and early out for both opposing values.
    if (depthSampleA > depthThresholdHigh || depthSampleA == 0.0)
        return;
    if (depthSampleB > depthThresholdHigh || depthSampleB == 0.0)
        return;
    // Is depth too low? Clamp to lower bound
    [[flatten]] if (depthSampleA < depthThresholdLow) depthSampleA = lowerDepthBound;
    [[flatten]] if (depthSampleB < depthThresholdLow) depthSampleB = lowerDepthBound;

    // Dynamic depth range.
    depthThresholdLow = min(depthThresholdLow, min(depthSampleB, depthSampleA) - depthThreshold);
    depthThresholdHigh = max(depthThresholdHigh, max(depthSampleB, depthSampleA) + depthThreshold);

    // Add samples
    totalWeight += gaussianWeight * 2.0;
    filteredDepth += (depthSampleA + depthSampleB) * gaussianWeight;
}

void main() {
    ivec2 screenCoord = ivec2(gl_GlobalInvocationID.xy);
    float centerDepth = texelFetch(DepthSource, screenCoord, 0).r;
    if (isinf(centerDepth) || centerDepth == 0.0) {
        return;
    }

    float sigma = imageSize(DepthDest).y * worldSpaceSigma / (Camera.TanHalfVerticalFov * centerDepth * 2.0);
    float filterSizef = min(MAX_FILTER_SIZE, sigma * SIGMA_TO_FILTERSIZE);
    sigma = filterSizef * (1.0 / SIGMA_TO_FILTERSIZE); // correct sigma so we don't degenerate to a box filter
    float gaussianK = 0.5 / sq(sigma);
    int filterSize = int(ceil(filterSizef));

    float filteredDepth = centerDepth;
    float totalWeight = 1.0;

    float depthThresholdHigh = centerDepth + depthThreshold;
    float depthThresholdLow = centerDepth - depthThreshold;
    float lowerDepthBound = centerDepth - mu;

#if defined(FILTER_2D)

    // Sample from middle to outside
    for (int r = 1; r < filterSize; ++r) {
        // Go round the square, sampling 4 equidistant points at the time (starting with the corners)
        for (int i = 0; i < r * 2; ++i) {
            float gaussianWeight = exp(-(sq(r) + sq(r - i)) * gaussianK);

            float depthA = texelFetch(DepthSource, screenCoord + ivec2(r, r - i), 0).r;
            float depthB = texelFetch(DepthSource, screenCoord - ivec2(r, r - i), 0).r;
            narrowRangeFilter(depthA, depthB, lowerDepthBound, gaussianWeight, depthThreshold, depthThresholdHigh, depthThresholdLow, totalWeight,
                              filteredDepth);

            depthA = texelFetch(DepthSource, screenCoord + ivec2(r - i, -r), 0).r;
            depthB = texelFetch(DepthSource, screenCoord - ivec2(r - i, -r), 0).r;
            narrowRangeFilter(depthA, depthB, lowerDepthBound, gaussianWeight, depthThreshold, depthThresholdHigh, depthThresholdLow, totalWeight,
                              filteredDepth);
        }
    }
#else
    for (int r = 1; r < filterSize; ++r) {
        float gaussianWeight = exp(-sq(r) * gaussianK);

        ivec2 offset = ivec2(0);
        offset[FilterDirection] = r;

        float depthA = texelFetch(DepthSource, screenCoord + offset, 0).r;
        float depthB = texelFetch(DepthSource, screenCoord - offset, 0).r;
        narrowRangeFilter(depthA, depthB, lowerDepthBound, gaussianWeight, depthThreshold, depthThresholdHigh, depthThresholdLow, totalWeight,
                          filteredDepth);
    }
#endif

    filteredDepth /= totalWeight;
    // filteredDepth = centerDepth;
    imageStore(DepthDest, screenCoord, filteredDepth.rrrr);
}