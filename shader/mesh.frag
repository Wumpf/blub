#version 460

#include "background.glsl"
#include "mesh.glsl"
#include "sh.glsl"
#include "utilities.glsl"

layout(location = 0) in vec3 in_Normal;
layout(location = 0) out vec4 out_Color;

void main() {
    vec3 normal = normalize((Meshes[MeshIndex].Transform * vec4(in_Normal, 0.0)).xyz);

    vec3 brdf = vec3(1.0) / PI;
    vec3 radiance = brdf * saturate(dot(normal, -DirectionalLightDirection)) * DirectionalLightRadiance;
    radiance += brdf * saturate(sh3EvaluateCosine(normal, IndirectRadianceSH3)) *
                4.0; // because everyone loves exagerated indirect light (need better tonemap!! TODO)
    out_Color = vec4(radiance, 1);
}