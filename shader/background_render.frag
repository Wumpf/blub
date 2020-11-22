#version 460

//#define VISUALIZE_SH_RADIANCE

#include "background.glsl"
#include "global_bindings.glsl"
#include "utilities.glsl"

#ifdef VISUALIZE_SH_RADIANCE
#include "sh.glsl"
#endif

layout(location = 0) out vec4 out_Color;
layout(depth_less) out float gl_FragDepth;

void main() {
    // Too lazy to do this cleaner, also doesn't matter perf wise :)
    vec3 dir = reconstructWorldPositionFromViewSpaceDepth(gl_FragCoord.xy * Screen.ResolutionInv, 1.0) - Camera.Position;

    out_Color.rgb = sampleBackground(Camera.Position, dir, gl_FragDepth);
    out_Color.a = 0.0; // We mark the background with alpha 0

#ifdef VISUALIZE_SH_RADIANCE
    dir = normalize(dir);
    out_Color.rgb = sh3Evaluate(dir, IndirectRadianceSH3);
#endif
}