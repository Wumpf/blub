#version 450

#include "fluid_render_info.glsl"
#include "global_bindings.glsl"
#include "sphere_particles.glsl"
#include "utilities.glsl"

out gl_PerVertex { vec4 gl_Position; };

layout(location = 0) out vec3 out_WorldPosition;
layout(location = 1) out vec3 out_ParticleWorldPosition;
layout(location = 2) out float out_Radius;

void main() {
    out_Radius = Rendering.FluidParticleRadius;
    out_ParticleWorldPosition = Particles[gl_InstanceIndex].Position * Rendering.FluidGridToWorldScale + Rendering.FluidWorldMin;
    out_WorldPosition = spanParticle(out_ParticleWorldPosition, out_Radius);
    gl_Position = Camera.ViewProjection * vec4(out_WorldPosition, 1.0);
}
