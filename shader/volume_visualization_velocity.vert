#version 450

#include "per_frame_resources.glsl"
#include "utilities.glsl"

layout(set = 1, binding = 1) uniform texture3D VelocityVolume;

out gl_PerVertex { vec4 gl_Position; };

layout(location = 0) out vec3 out_Color;

void main() {
    ivec3 volumeSize = textureSize(VelocityVolume, 0);

    uint channel = gl_InstanceIndex % 3;
    uint positionIndex = gl_InstanceIndex / 3;
    ivec3 volumeCoordinate =
        ivec3(positionIndex % volumeSize.x, positionIndex / volumeSize.x % volumeSize.y, positionIndex / volumeSize.x / volumeSize.y);

    vec3 cellCenter = volumeCoordinate + vec3(0.5);
    vec3 linePosition = cellCenter;
    linePosition[channel] += 0.5;

    vec3 velocity = texelFetch(VelocityVolume, volumeCoordinate, 0).xyz;
    float scale = clamp(velocity[channel] * Rendering.VelocityVisualizationScale, -1.0, 1.0);
    if (gl_VertexIndex == 0) {
        linePosition[channel] += scale;
    }

    out_Color = colormapCoolToWarm(scale);
    gl_Position = Camera.ViewProjection * vec4(linePosition, 1.0);
}
