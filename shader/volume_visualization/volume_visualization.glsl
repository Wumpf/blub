#include "global_bindings.glsl"

ivec3 getVolumeCoordinate(uint positionIndex) {
    return ivec3(positionIndex % Rendering.FluidGridResolution.x, positionIndex / Rendering.FluidGridResolution.x % Rendering.FluidGridResolution.y,
                 positionIndex / Rendering.FluidGridResolution.x / Rendering.FluidGridResolution.y);
}
