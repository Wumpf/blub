#include "per_frame_resources.glsl"
#include "sphere_particles.glsl"
#include "utilities.glsl"

layout(set = 1, binding = 1) uniform texture3D VelocityVolume;
layout(set = 1, binding = 2) uniform texture3D DivergenceVolume;
layout(set = 1, binding = 3) uniform texture3D PressureVolume;

out gl_PerVertex { vec4 gl_Position; };

layout(location = 0) out vec3 out_WorldPosition;
layout(location = 1) out vec3 out_ParticleWorldPosition;
layout(location = 2) out vec3 out_Tint;
layout(location = 3) out float out_Radius;

void main() {
    ivec3 volumeSize = textureSize(VelocityVolume, 0);

    ivec3 volumeCoordinate =
        ivec3(gl_InstanceIndex % volumeSize.x, gl_InstanceIndex / volumeSize.x % volumeSize.y, gl_InstanceIndex / volumeSize.x / volumeSize.y);

#if defined(VISUALIZE_DIVERGENCE)
    float divergence = texelFetch(DivergenceVolume, volumeCoordinate, 0).x;
    float scale = saturate(sq(divergence) * 0.8);
    out_Tint = heatmapColor(scale).bgr;
#elif defined(VISUALIZE_PRESSURE)
    float pressure = texelFetch(PressureVolume, volumeCoordinate, 0).x;
    float scale = saturate(abs(pressure) * 0.5);
    out_Tint = heatmapColor(scale).grb;
#endif

    out_ParticleWorldPosition = volumeCoordinate + vec3(0.5);
    out_Radius = scale * 0.5;
    out_WorldPosition = spanParticle(out_ParticleWorldPosition, out_Radius);
    gl_Position = Camera.ViewProjection * vec4(out_WorldPosition, 1.0);
}
