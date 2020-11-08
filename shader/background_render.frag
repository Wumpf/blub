#version 460

//#define VISUALIZE_SH_RADIANCE

#include "background.glsl"
#include "per_frame_resources.glsl"
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
    out_Color.a = 1.0;

#ifdef VISUALIZE_SH_RADIANCE
    vec3 shCoeffs[9] =
        vec3[9](vec3(0.26880783, 0.26525503, 0.25323117), vec3(-0.18061855, -0.182183, -0.17731333), vec3(0.13950694, 0.13673073, 0.12422592),
                vec3(-0.3194243, -0.31284282, -0.28476116), vec3(0.20926912, 0.20343526, 0.1833044), vec3(-0.09322964, -0.09043422, -0.08140675),
                vec3(-0.09613961, -0.0930359, -0.08279946), vec3(-0.15792988, -0.15315816, -0.13775906), vec3(0.1190775, 0.11848518, 0.1101599));
    dir = normalize(dir);
    out_Color.rgb = sh3Evaluate(dir, shCoeffs);
#endif
}