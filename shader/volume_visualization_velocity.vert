#version 450

#include "per_frame_resources.glsl"
#include "simulation/hybrid_fluid.glsl"
#include "utilities.glsl"
#include "visualization.glsl"

out gl_PerVertex { vec4 gl_Position; };

layout(location = 0) out vec3 out_Color;

void main() {
    ivec3 volumeCoordinate = getVolumeCoordinate(gl_InstanceIndex / 3);
    uint channel = gl_InstanceIndex % 3;

    uint marker = texelFetch(MarkerVolume, volumeCoordinate, 0).x;

    vec3 cellCenter = volumeCoordinate + vec3(0.5);
    vec3 linePosition = cellCenter;
    linePosition[channel] += 0.5;

    vec3 velocity = texelFetch(VelocityVolume, volumeCoordinate, 0).xyz;
    float scale = marker == CELL_FLUID ? clamp(velocity[channel] * Rendering.VelocityVisualizationScale, -1.0, 1.0) : 0.0;
    if (gl_VertexIndex == 0) {
        linePosition[channel] += scale;
    }

    out_Color = colormapCoolToWarm(scale);
    gl_Position = Camera.ViewProjection * vec4(linePosition, 1.0);
}
