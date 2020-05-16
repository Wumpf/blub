#version 450

#include "per_frame_resources.glsl"
#include "utilities.glsl"

layout(set = 1, binding = 1) uniform texture3D VelocityVolume;

out gl_PerVertex { vec4 gl_Position; };

layout(location = 0) out vec3 out_Color;

void main() {
    ivec3 volumeSize = textureSize(VelocityVolume, 0);

    ivec3 volumeCoordinate =
        ivec3(gl_InstanceIndex % volumeSize.x, gl_InstanceIndex / volumeSize.x % volumeSize.y, gl_InstanceIndex / volumeSize.x / volumeSize.y);

    vec3 cellCenter = volumeCoordinate + vec3(0.5);
    vec3 linePosition = cellCenter;

    vec3 velocity = texelFetch(VelocityVolume, volumeCoordinate, 0).xyz;
    float velocityMagnitude = length(velocity);
    float scale = min(velocityMagnitude * Rendering.VelocityVisualizationScale, 1.0) / velocityMagnitude;
    if (gl_VertexIndex == 0) {
        linePosition += velocity * scale;
    }

    out_Color = colormapHeat(velocityMagnitude * Rendering.VelocityVisualizationScale);
    gl_Position = Camera.ViewProjection * vec4(linePosition, 1.0);
}
