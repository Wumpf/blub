#version 460

#include "../global_bindings.glsl"
#include "../utilities.glsl"

layout(location = 0) in vec3 in_WorldPosition;
layout(location = 1) in vec3 in_ParticleWorldPosition;
layout(location = 2) in float in_Radius;
layout(location = 0) out float out_ViewSpaceDepth;
layout(location = 1) out float out_Thickness;

void main() {
    vec3 rayDir = normalize(in_WorldPosition - Camera.Position);
    float cameraDistance;
    float cameraDistanceFar;
    // TODO: Elipsoids using APIC matrix?
    if (!sphereIntersect(in_ParticleWorldPosition, in_Radius, Camera.Position, rayDir, cameraDistance, cameraDistanceFar))
        discard;

    vec3 cameraPosToSpherePos = cameraDistance * rayDir;

    out_ViewSpaceDepth = dot(Camera.Direction, cameraPosToSpherePos);
    // quadratic splats. Compensate a bit for particle overlap
    out_Thickness = (cameraDistanceFar - cameraDistance) * (0.25 * Rendering.FluidGridToWorldScale / Rendering.FluidParticleRadius);
}
