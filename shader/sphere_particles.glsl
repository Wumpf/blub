#include "global_bindings.glsl"
#include "utilities.glsl"

const vec2 quadPositions[4] = vec2[4](vec2(-1.0, -1.0), vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0));

vec3 spanParticle(vec3 particleCenter, float radius) {
    // Spanning billboards is easy!
    vec3 toCamera = Camera.Position - particleCenter;
    float distanceToCameraSq = dot(toCamera, toCamera);
    float distanceToCameraInv = inversesqrt(distanceToCameraSq);
    vec3 particleNormal = toCamera * distanceToCameraInv;
    vec3 particleRight = normalize(cross(particleNormal, Camera.Up)); // It's spheres so any orthogonal vector would do.
    vec3 particleUp = cross(particleRight, particleNormal);
    vec3 quadPosition = (quadPositions[gl_VertexIndex].x * particleRight + quadPositions[gl_VertexIndex].y * particleUp);

    // But we want to simulate spheres here!
    // If camera gets close to a sphere (or the sphere is large) then outlines of the sphere would not fit on a quad with radius r!
    // Enlarging the quad is one solution, but then Z gets tricky (== we need to write correct Z and not quad Z to depth buffer) since we may get
    // "unnecessary" overlaps. So instead, we change the size _and_ move the sphere closer (using math!)
    float cameraOffset = radius * radius * distanceToCameraInv;
    float modifiedRadius = radius * distanceToCameraInv * sqrt(distanceToCameraSq - radius * radius);
    return particleCenter + quadPosition * modifiedRadius + cameraOffset * particleNormal;

    // normal billboard (spheres are cut off)
    // return particleCenter + quadPosition * radius;

    // only enlarged billboard (works but requires z care even for non-overlapping spheres)
    // modifiedRadius = length(toCamera) * radius / sqrt(distanceToCameraSq - radius * radius);
    // return particleCenter + quadPosition * modifiedRadius;
}
