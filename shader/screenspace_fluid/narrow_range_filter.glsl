// An implementation of:
// A Narrow-Range Filter for Screen-Space Fluid Rendering, Truong et al. 2018
// http://www.cemyuksel.com/research/papers/narrowrangefilter.pdf

// With some flavoring of my own.

#if !defined(FILTER_1D) && !defined(FILTER_2D)
#error "Need to define either FILTER_1D or FILTER_2D"
#endif

// Total filter size is (HALF_MAX_FILTER_SIZE + HALF_MAX_FILTER_SIZE + 1)
#if defined(FILTER_2D)

// We fix the filter size to less than half the local size
// This simplifies the shader, as every thread is responsible for four samples now!
#define HALF_MAX_FILTER_SIZE 6 // Careful, has very strong effect on performance.
#define LOCAL_SIZE 16
layout(local_size_x = LOCAL_SIZE, local_size_y = LOCAL_SIZE, local_size_z = 1) in;

// has generous padding to avoid bank conflicts (if HALF_MAX_FILTER_SIZE is less than LOCAL_SIZE/2)
shared float sharedBuffer[LOCAL_SIZE * 2][LOCAL_SIZE * 2];

#else

#define LOCAL_SIZE 64
layout(local_size_x = LOCAL_SIZE, local_size_y = 1, local_size_z = 1) in;

// We fix the filter size to half the local size
// This simplifies the shader, as every thread is responsible for two samples now!
#define HALF_MAX_FILTER_SIZE (LOCAL_SIZE / 2)
shared float sharedBuffer[LOCAL_SIZE * 2];

#endif

#include "screenspace_fluid/filter.glsl"

#include "fluid_render_info.glsl"
#include "global_bindings.glsl"
#include "utilities.glsl"

layout(set = 2, binding = 0, r32f) uniform restrict writeonly image2D DepthDest;
layout(set = 2, binding = 1) uniform texture2D DepthSource;

// Relationship between the standard deviation sigma (for gaussian kernel) and the filter size.
// (the lower, the boxier the filter gets)
#define SIGMA_TO_FILTERSIZE 3.0

// TODO: Expose
float worldSpaceSigma = 1.5 * Rendering.FluidParticleRadius;
float depthThreshold = 10.0 * Rendering.FluidParticleRadius;
float mu = 1.0 * Rendering.FluidParticleRadius;

// depthSamples are depth samples on opposing sides, both at the same distance to the middle.
void narrowRangeFilter(vec2 depthSamples, float higherDepthBound, float gaussianWeight, float depthThreshold, inout float depthThresholdHigh,
                       inout float depthThresholdLow, inout float totalWeight, inout float filteredDepth) {
    // Is depth too low? Keep filter symmetric and early out for both opposing values.
    if (any(lessThan(depthSamples, vec2(depthThresholdLow))))
        return;

    // Is depth too high? Clamp to upper bound.
    [[flatten]] if (depthSamples.x > depthThresholdHigh) depthSamples.x = higherDepthBound;
    [[flatten]] if (depthSamples.y > depthThresholdHigh) depthSamples.y = higherDepthBound;

    // Dynamic depth range.
    depthThresholdLow = min(depthThresholdLow, min(depthSamples.x, depthSamples.y) - depthThreshold);
    depthThresholdHigh = max(depthThresholdHigh, max(depthSamples.x, depthSamples.y) + depthThreshold);

    // Add samples
    totalWeight += gaussianWeight * 2.0;
    filteredDepth += (depthSamples.x + depthSamples.y) * gaussianWeight;
}

void main() {
    uvec2 screenCoord = getScreenCoord();
    uvec2 screenSize = imageSize(DepthDest);
#if defined(FILTER_1D)

    // Preload to shared memory.
    // Filter size is known to be half the local size -> 2 samples
    {
        uvec2 sampleCoord = addInFilterDirection(screenCoord, -HALF_MAX_FILTER_SIZE);
        sharedBuffer[gl_LocalInvocationID.x] = texelFetch(DepthSource, ivec2(sampleCoord), 0).r;
        sampleCoord = addInFilterDirection(screenCoord, HALF_MAX_FILTER_SIZE);
        sharedBuffer[gl_LocalInvocationID.x + LOCAL_SIZE] = texelFetch(DepthSource, ivec2(sampleCoord), 0).r;
        barrier();
    }

    const uint sharedBufferCenterIndex = gl_LocalInvocationID.x + HALF_MAX_FILTER_SIZE;
    float centerDepth = sharedBuffer[sharedBufferCenterIndex];
#else

    // Preload to shared memory.
    // Filter size is known to be half the local size -> 4 samples
    {
        // Using textureGather we can do all this with a single sample!
        // .. but any winings we have in that version are undone with the resulting 2x bank conflict on smem write!
#if 1
        const uvec2 sampleCoordMin = screenCoord - uvec2(HALF_MAX_FILTER_SIZE);
        const uvec2 sampleCoordMax = screenCoord + uvec2(HALF_MAX_FILTER_SIZE);
        const uvec2 smemIndexMin = gl_LocalInvocationID.xy;
        const uvec2 smemIndexMax = gl_LocalInvocationID.xy + uvec2(HALF_MAX_FILTER_SIZE * 2);
        sharedBuffer[smemIndexMin.y][smemIndexMin.x] = texelFetch(DepthSource, ivec2(sampleCoordMin), 0).r;
        if (smemIndexMax.x < LOCAL_SIZE + HALF_MAX_FILTER_SIZE * 2 && smemIndexMax.y < LOCAL_SIZE + HALF_MAX_FILTER_SIZE * 2) {
            sharedBuffer[smemIndexMin.y][smemIndexMax.x] = texelFetch(DepthSource, ivec2(sampleCoordMax.x, sampleCoordMin.y), 0).r;
            sharedBuffer[smemIndexMax.y][smemIndexMin.x] = texelFetch(DepthSource, ivec2(sampleCoordMin.x, sampleCoordMax.y), 0).r;
            sharedBuffer[smemIndexMax.y][smemIndexMax.x] = texelFetch(DepthSource, ivec2(sampleCoordMax), 0).r;
        }
#else
        vec2 gatherCoord =
            (vec2(getBlockScreenCoord() - uvec2(HALF_MAX_FILTER_SIZE) + gl_LocalInvocationID.xy * 2) + vec2(0.5)) / (screenSize - uvec2(1));
        vec4 samples = vec4(0);
        if (all(greaterThanEqual(gatherCoord, vec2(0))) && all(lessThanEqual(gatherCoord, vec2(1.0)))) // todo: Can we have a clamp to zero sampler?
            samples = textureGather(sampler2D(DepthSource, SamplerPointClamp), gatherCoord);
        const uvec2 smemIndexMin = gl_LocalInvocationID.xy * 2;
        sharedBuffer[smemIndexMin.y + 1][smemIndexMin.x + 0] = samples.x;
        sharedBuffer[smemIndexMin.y + 1][smemIndexMin.x + 1] = samples.y;
        sharedBuffer[smemIndexMin.y + 0][smemIndexMin.x + 1] = samples.z;
        sharedBuffer[smemIndexMin.y + 0][smemIndexMin.x + 0] = samples.w;
#endif
        barrier();
    }

    const uvec2 sharedBufferCenterIndex = gl_LocalInvocationID.xy + uvec2(HALF_MAX_FILTER_SIZE);
    float centerDepth = sharedBuffer[sharedBufferCenterIndex.y][sharedBufferCenterIndex.x];
#endif
    if (centerDepth > 9999.0 || centerDepth == 0.0) {
        return;
    }

    float sigma = screenSize.y * worldSpaceSigma / (Camera.TanHalfVerticalFov * centerDepth * 2.0);
    float filterSizef = min(HALF_MAX_FILTER_SIZE, sigma * SIGMA_TO_FILTERSIZE);
    sigma = filterSizef * (1.0 / SIGMA_TO_FILTERSIZE); // correct sigma so we don't degenerate to a box filter
    const float gaussianK = 0.5 / sq(sigma);
    uint filterSize = uint(ceil(filterSizef));

    float filteredDepth = centerDepth;
    float totalWeight = 1.0;

    float depthThresholdHigh = centerDepth + depthThreshold;
    float depthThresholdLow = centerDepth - depthThreshold;
    const float higherDepthBound = centerDepth + mu;

#if defined(FILTER_2D)
    // Sample from middle to outside
    for (uint r = 1; r <= filterSize; ++r) {
        // Go round the square, sampling 4 equidistant points at the time (starting with the corners)
        for (uint i = 0; i < r * 2; ++i) {
            float gaussianWeight = exp(-(sq(r) + sq(r - i)) * gaussianK);

            vec2 depthSamples;
            depthSamples.x = sharedBuffer[sharedBufferCenterIndex.y + (r - i)][sharedBufferCenterIndex.x + r];
            depthSamples.y = sharedBuffer[sharedBufferCenterIndex.y - (r - i)][sharedBufferCenterIndex.x - r];
            narrowRangeFilter(depthSamples, higherDepthBound, gaussianWeight, depthThreshold, depthThresholdHigh, depthThresholdLow, totalWeight,
                              filteredDepth);
            depthSamples.x = sharedBuffer[sharedBufferCenterIndex.y - r][sharedBufferCenterIndex.x + (r - i)];
            depthSamples.y = sharedBuffer[sharedBufferCenterIndex.y + r][sharedBufferCenterIndex.x - (r - i)];
            narrowRangeFilter(depthSamples, higherDepthBound, gaussianWeight, depthThreshold, depthThresholdHigh, depthThresholdLow, totalWeight,
                              filteredDepth);
        }
    }
#else
    for (uint r = 1; r <= filterSize; ++r) {
        float gaussianWeight = exp(-sq(r) * gaussianK);

        vec2 depthSamples;
        depthSamples.x = sharedBuffer[sharedBufferCenterIndex - r];
        depthSamples.y = sharedBuffer[sharedBufferCenterIndex + r];
        narrowRangeFilter(depthSamples, higherDepthBound, gaussianWeight, depthThreshold, depthThresholdHigh, depthThresholdLow, totalWeight,
                          filteredDepth);
    }
#endif

    filteredDepth /= totalWeight;
    // filteredDepth = centerDepth;
    imageStore(DepthDest, ivec2(screenCoord), filteredDepth.rrrr);
}