#version 460

#include "background.glsl"
#include "global_bindings.glsl"
#include "sh.glsl"
#include "utilities.glsl"

layout(location = 0) in vec3 in_WorldPosition;
layout(location = 1) in flat ivec3 in_VolumeCoordinate;
layout(location = 0) out vec4 out_Color;

layout(set = 2, binding = 0) uniform texture3D SceneVoxelization;

void main() {
    vec3 dxX = dFdx(in_WorldPosition);
    vec3 dxY = dFdy(in_WorldPosition);
    vec3 normal = normalize(cross(dxY, dxX));

    vec3 voxelSpeed = abs(texelFetch(SceneVoxelization, in_VolumeCoordinate, 0).xyz);
    vec3 brdf = voxelSpeed * Rendering.VelocityVisualizationScale;

    vec3 radiance = brdf * saturate(dot(normal, -DirectionalLightDirection)) * DirectionalLightRadiance;
    radiance += brdf * saturate(sh3EvaluateCosine(normal, IndirectRadianceSH3)) * 4.0;
    out_Color = vec4(radiance, 1);
}