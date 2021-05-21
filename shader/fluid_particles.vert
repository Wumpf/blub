#version 450

#include "fluid_render_info.glsl"
#include "global_bindings.glsl"
#include "sphere_particles.glsl"
#include "utilities.glsl"

out gl_PerVertex { vec4 gl_Position; };

layout(push_constant) uniform PushConstants { uint VisualizationType; };

#define VISUALIZE_VELOCITY 0
#define VISUALIZE_INDEX 1

layout(location = 0) out vec3 out_WorldPosition;
layout(location = 1) out vec3 out_ParticleWorldPosition;
layout(location = 2) out vec3 out_Tint;
layout(location = 3) out float out_Radius;

void main() {
    out_Radius = Rendering.FluidParticleRadius;

    switch (VisualizationType) {
    case VISUALIZE_VELOCITY: {
        vec3 velocity = vec3(ParticleBufferVelocityX[gl_InstanceIndex].w, ParticleBufferVelocityY[gl_InstanceIndex].w,
                             ParticleBufferVelocityZ[gl_InstanceIndex].w);
        out_Tint = colormapHeat(length(velocity) * Rendering.VelocityVisualizationScale);
        break;
    }
    case VISUALIZE_INDEX:
        out_Tint = vec3(fract(gl_InstanceIndex / 255.0));
        break;
    }

    out_ParticleWorldPosition = Particles[gl_InstanceIndex].Position * Rendering.FluidGridToWorldScale + Rendering.FluidWorldMin;
    out_WorldPosition = spanParticle(out_ParticleWorldPosition, out_Radius);
    gl_Position = Camera.ViewProjection * vec4(out_WorldPosition, 1.0);
}
