#version 450

#include "global_bindings.glsl"
#include "utilities.glsl"

layout(location = 0) in vec3 in_WorldPosition;
layout(location = 1) in vec3 in_ParticleWorldPosition;
layout(location = 2) in vec3 in_Tint;
layout(location = 3) in float in_Radius;
layout(location = 0) out vec4 out_Color;

// Note that we promise to only lessen the depth value, so gpu can still do some hi-z/early depth culling
layout(depth_less) out float gl_FragDepth;

void main() {
    const vec3 lightdir = normalize(vec3(1.0, 2.0, 1.0));

    vec3 rayDir = normalize(in_WorldPosition - Camera.Position);
    float cameraDistance;
    if (!sphereIntersect(in_ParticleWorldPosition, in_Radius, Camera.Position, rayDir, cameraDistance))
        discard;

    vec3 sphereWorldPos = Camera.Position + cameraDistance * rayDir;
    vec3 normal = (sphereWorldPos - in_ParticleWorldPosition) / in_Radius;

    // Adjust depth buffer value.
    vec2 projected_zw = (Camera.ViewProjection * vec4(sphereWorldPos, 1.0)).zw; // (trusting optimizer to pick the right thing ;-))
    gl_FragDepth = projected_zw.x / projected_zw.y;

    // via https://www.iquilezles.org/www/articles/outdoorslighting/outdoorslighting.htm
    float sun = saturate(dot(lightdir, normal));
    float sky = saturate(0.5 + 0.5 * normal.y);
    float ind = saturate(dot(normal, normalize(lightdir * vec3(-1.0, 0.0, -1.0))));
    vec3 lighting = sun * vec3(1.64, 1.27, 0.99);
    lighting += sky * vec3(0.16, 0.20, 0.28);
    lighting += ind * vec3(0.40, 0.28, 0.20);

    out_Color = vec4(lighting * max(vec3(0.05), in_Tint), 1.0);
}
