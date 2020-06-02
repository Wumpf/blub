#version 450

#include "per_frame_resources.glsl"
#include "simulation/hybrid_fluid.glsl"
#include "utilities.glsl"
#include "visualization.glsl"

out gl_PerVertex { vec4 gl_Position; };

layout(location = 0) out vec4 out_Color;

float getChannel(vec3 v, uint channel) {
    switch (channel) {
    case 0:
        return v.x;
    case 1:
        return v.y;
    default:
        return v.z;
    }
}

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

    uint marker = texelFetch(MarkerVolume, volumeCoordinate, 0).x;

    vec3 cellCenter = (volumeCoordinate + vec3(0.5)) * Rendering.FluidGridToWorldScale + Rendering.FluidWorldOrigin;
    vec3 linePosition = cellCenter;
    addToChannel(linePosition, 0.5 * Rendering.FluidGridToWorldScale, channel);

    float velocity = getChannel(texelFetch(VelocityVolume, volumeCoordinate, 0).xyz, channel);
    float scale = clamp(velocity * Rendering.VelocityVisualizationScale, -1.0, 1.0);
    if (marker != CELL_FLUID)
        scale = 0.0;
    if (gl_VertexIndex == 0) {
        addToChannel(linePosition, scale * Rendering.FluidGridToWorldScale, channel);
    }

    out_Color = vec4(colormapCoolToWarm(scale), 1.0);
    gl_Position = Camera.ViewProjection * vec4(linePosition, 1.0);
}
