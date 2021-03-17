#version 460

#include "background.glsl"
#include "global_bindings.glsl"
#include "sh.glsl"
#include "utilities.glsl"

layout(location = 0) in vec3 in_WorldPosition;
layout(location = 0) out vec4 out_Color;

void main() {
    vec3 dxX = dFdx(in_WorldPosition);
    vec3 dxY = dFdy(in_WorldPosition);
    vec3 normal = normalize(cross(dxY, dxX));

    vec3 albedo = vec3(0.1);
    vec3 brdf = albedo / PI;

    vec3 radiance = brdf * saturate(dot(normal, -DirectionalLightDirection)) * DirectionalLightRadiance;
    radiance += brdf * saturate(sh3EvaluateCosine(normal, IndirectRadianceSH3)) * 4.0;
    out_Color = vec4(radiance, 1);
}