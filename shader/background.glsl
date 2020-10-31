#include "per_frame_resources.glsl"
#include "utilities.glsl"

layout(set = 1, binding = 0) uniform BackgroundAndLighting {
    vec3 DirectionalLightDirection;
    vec3 DirectionalLightRadiance;
};
layout(set = 1, binding = 1) uniform textureCube CubemapRgbe;

vec3 decodeRGBE(vec4 hdr) { return hdr.rgb * exp2((hdr.a * 255.0) - 128.0); }

vec3 sampleHdrCubemap(vec3 dir) {
    vec4 rgbe = texture(samplerCube(CubemapRgbe, SamplerTrilinearClamp), dir);
    return decodeRGBE(rgbe);
}
