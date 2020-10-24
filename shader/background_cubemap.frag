#version 460

#include "per_frame_resources.glsl"
#include "sky.glsl"
#include "utilities.glsl"

layout(location = 0) out vec4 out_Color;

void main() {
    // Too lazy to do this cleaner, also doesn't matter perf wise :)
    vec3 dir = reconstructWorldPositionFromViewSpaceDepth(gl_FragCoord.xy * Screen.ResolutionInv, 1.0) - Camera.Position;

    out_Color.xyz = sampleHdrCubemap(dir);
    out_Color.w = 1.0;
}