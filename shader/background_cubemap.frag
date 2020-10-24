#version 460

#include "per_frame_resources.glsl"
#include "utilities.glsl"

layout(set = 1, binding = 0) uniform textureCube Cubemap;

layout(location = 0) out vec4 out_Color;

void main() {
    // Too lazy to do this cleaner, also doesn't matter perf wise :)
    vec3 dir = reconstructWorldPositionFromViewSpaceDepth(gl_FragCoord.xy * Screen.ResolutionInv, 1.0) - Camera.Position;

    vec4 rgbe = texture(samplerCube(Cubemap, SamplerTrilinearClamp), dir);
    vec3 hdr_rgb = decodeRGBE(rgbe);

    const float exposure = 1.5;
    out_Color.xyz = hdr_rgb * exposure;
    out_Color.w = 1.0;
}