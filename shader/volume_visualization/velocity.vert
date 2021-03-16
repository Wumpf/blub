#version 450

#define NO_SIMPROPS

#include "../fluid_render_info.glsl"
#include "../global_bindings.glsl"
#include "../simulation/hybrid_fluid.glsl"
#include "../utilities.glsl"
#include "volume_visualization.glsl"

out gl_PerVertex { vec4 gl_Position; };

layout(location = 0) out vec4 out_Color;

void addToChannel(inout vec3 v, float value, uint channel) {
    switch (channel) {
    case 0:
        v.x += value;
        break;
    case 1:
        v.y += value;
        break;
    default:
        v.z += value;
        break;
    }
}

void main() {
    ivec3 volumeCoordinate = getVolumeCoordinate(gl_InstanceIndex / 3);
    uint channel = gl_InstanceIndex % 3;

    float marker = texelFetch(MarkerVolume, volumeCoordinate, 0).x;

    vec3 cellCenter = (volumeCoordinate + vec3(0.5)) * Rendering.FluidGridToWorldScale + Rendering.FluidWorldMin;
    vec3 linePosition = cellCenter;
    addToChannel(linePosition, 0.5 * Rendering.FluidGridToWorldScale, channel);

    float velocity = 0.0;
    float neighborMarker = CELL_SOLID;
    switch (channel) {
    case 0:
        neighborMarker = texelFetch(MarkerVolume, volumeCoordinate + ivec3(1, 0, 0), 0).x;
        velocity = texelFetch(VelocityVolumeX, volumeCoordinate, 0).x;
        break;
    case 1:
        neighborMarker = texelFetch(MarkerVolume, volumeCoordinate + ivec3(0, 1, 0), 0).x;
        velocity = texelFetch(VelocityVolumeY, volumeCoordinate, 0).x;
        break;
    default:
        neighborMarker = texelFetch(MarkerVolume, volumeCoordinate + ivec3(0, 0, 1), 0).x;
        velocity = texelFetch(VelocityVolumeZ, volumeCoordinate, 0).x;
        break;
    }

    float scale = clamp(velocity * Rendering.VelocityVisualizationScale, -1.0, 1.0);
    if (marker != CELL_FLUID && neighborMarker != CELL_FLUID)
        scale = 0.0; // Showing extrapolated velocities is a bit complicated.
    // For debugging it can be useful to fill the velocity field with NaN and see what remains.
    if (isnan(velocity))
        scale = 0.0;
    if (gl_VertexIndex == 0) {
        addToChannel(linePosition, scale * Rendering.FluidGridToWorldScale, channel);
    }

    out_Color = vec4(colormapCoolToWarm(scale), 1.0);
    gl_Position = Camera.ViewProjection * vec4(linePosition, 1.0);
}
