#version 450

#include "per_frame_resources.glsl"
#include "sphere_particles.glsl"
#include "utilities.glsl"
#include "visualization.glsl"

out gl_PerVertex { vec4 gl_Position; };

layout(location = 0) out vec3 out_WorldPosition;
layout(location = 1) out vec3 out_ParticleWorldPosition;
layout(location = 2) out vec3 out_Tint;
layout(location = 3) out float out_Radius;

void main() {
    out_Radius = 0.25; // todo, make scale dependent
    vec3 velocity = vec3(Particles[gl_InstanceIndex].VelocityMatrix[0].w, Particles[gl_InstanceIndex].VelocityMatrix[1].w,
                         Particles[gl_InstanceIndex].VelocityMatrix[2].w);
    out_Tint = colormapHeat(length(velocity) * Rendering.VelocityVisualizationScale);
    out_ParticleWorldPosition = Particles[gl_InstanceIndex].Position;
    out_WorldPosition = spanParticle(out_ParticleWorldPosition, out_Radius);
    gl_Position = Camera.ViewProjection * vec4(out_WorldPosition, 1.0);
}
