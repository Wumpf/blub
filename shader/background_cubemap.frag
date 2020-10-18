#version 460

#include "per_frame_resources.glsl"
#include "utilities.glsl"

layout(set = 1, binding = 0) uniform textureCube Cubemap;

layout(location = 0) out vec4 out_Color;

void main() {
    // Too lazy to do this cleaner, also doesn't matter perf wise :)
    vec3 dir = reconstructWorldPositionFromViewSpaceDepth(gl_FragCoord.xy * Screen.ResolutionInv, 1.0) - Camera.Position;
    out_Color = vec4(0.1, 0.2, 0.3, 1.0); //texture(samplerCube(Cubemap, SamplerTrilinearClamp), dir);
}