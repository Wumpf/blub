#version 450

#include "particles.glsl"
#include "per_frame_resources.glsl"
#include "utilities.glsl"

layout(set = 1, binding = 0, row_major) buffer restrict ParticleBuffer { Particle Particles[]; };

out gl_PerVertex { vec4 gl_Position; };

layout(location = 0) out vec3 out_WorldPosition;
layout(location = 1) out vec3 out_ParticleWorldPosition;
layout(location = 2) out vec3 out_Tint;

const vec2 quadPositions[4] = vec2[4](vec2(-1.0, -1.0), vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0));

vec3 heatmapColor(float t) { return saturate(vec3(t * 3, t * 3 - 1, t * 3 - 2)); }

void main() {
    const float radius = 0.25; // todo.
    vec3 velocity = Particles[gl_InstanceIndex].VelocityMatrix[3];
    out_Tint = heatmapColor(length(velocity) * 0.05); // TODO: Please make me tweakable

    out_ParticleWorldPosition = Particles[gl_InstanceIndex].Position;

    // Spanning billboards is easy!
    vec3 toCamera = Camera.Position - out_ParticleWorldPosition;
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
    out_WorldPosition = out_ParticleWorldPosition + quadPosition * modifiedRadius + cameraOffset * particleNormal;

    // normal billboard (spheres are cut off)
    // out_WorldPosition = out_ParticleWorldPosition + quadPosition * radius;

    // only enlarged billboard (works but requires z care even for non-overlapping spheres)
    // modifiedRadius = length(toCamera) * radius / sqrt(distanceToCameraSq - radius * radius);
    // out_WorldPosition = out_ParticleWorldPosition + quadPosition * modifiedRadius;

    gl_Position = Camera.ViewProjection * vec4(out_WorldPosition, 1.0);
}
