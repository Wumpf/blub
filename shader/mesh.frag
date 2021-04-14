#version 460

#include "background.glsl"
#include "sh.glsl"
#include "utilities.glsl"

layout(push_constant) uniform PushConstants_ { uint MeshIndex; };

layout(location = 0) in vec3 in_Normal;
layout(location = 1) in vec2 in_Texcoord;
layout(location = 0) out vec4 out_Color;

void main() {
    vec3 normal = normalize(vec4(in_Normal, 0.0) * Meshes[MeshIndex].WorldTransform);

    vec3 albedo = vec3(1.0);
    int textureIndex = Meshes[MeshIndex].TextureIndex;
    if (textureIndex >= 0) {
        albedo = texture(sampler2D(MeshTextures[textureIndex], SamplerTrilinearClamp), in_Texcoord).rgb;
    }

    vec3 brdf = albedo / PI;

    vec3 radiance = brdf * saturate(dot(normal, -DirectionalLightDirection)) * DirectionalLightRadiance;
    radiance += brdf * saturate(sh3EvaluateCosine(normal, IndirectRadianceSH3)) *
                4.0; // because everyone loves exagerated indirect light (need better tonemap!! TODO)
    out_Color = vec4(radiance, 1);
}