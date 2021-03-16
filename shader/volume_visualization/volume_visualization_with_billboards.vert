#version 450

#define NO_SIMPROPS

#include "fluid_render_info.glsl"
#include "simulation/hybrid_fluid.glsl"
#include "sphere_particles.glsl"
#include "utilities.glsl"
#include "volume_visualization.glsl"

out gl_PerVertex { vec4 gl_Position; };

layout(location = 0) out vec3 out_WorldPosition;
layout(location = 1) out vec3 out_ParticleWorldPosition;
layout(location = 2) out vec3 out_Tint;
layout(location = 3) out float out_Radius;

layout(push_constant) uniform PushConstants { uint VisualizationType; };

#define VISUALIZE_DIVERGENCE 0
#define VISUALIZE_PRESSURE_VELOCITY 1
#define VISUALIZE_PRESSURE_DENSITY 2
#define VISUALIZE_MARKER 3
#define VISUALIZE_DEBUG 4

float computeDivergenceForDirection(ivec3 coord, texture3D velocityVolume, float oppositeWallType, const uint component) {
    ivec3 neighborCoord = coord;
    neighborCoord[component] -= 1;

    if (oppositeWallType == CELL_FLUID)
        return texelFetch(velocityVolume, coord, 0).x - texelFetch(velocityVolume, neighborCoord, 0).x;
    else if (oppositeWallType == CELL_SOLID)
        return texelFetch(velocityVolume, coord, 0).x;
    else
        return 0.0;
}

void main() {
    ivec3 volumeCoordinate = getVolumeCoordinate(gl_InstanceIndex);
    float marker = texelFetch(MarkerVolume, volumeCoordinate, 0).x;
    float scale;

    switch (VisualizationType) {
    case VISUALIZE_DIVERGENCE:
        float divergence = 0.0;
        if (marker == CELL_FLUID) {
            float markerX0 = texelFetch(MarkerVolume, volumeCoordinate - ivec3(1, 0, 0), 0).x;
            divergence += computeDivergenceForDirection(volumeCoordinate, VelocityVolumeX, markerX0, 0);
            float markerY0 = texelFetch(MarkerVolume, volumeCoordinate - ivec3(0, 1, 0), 0).x;
            divergence += computeDivergenceForDirection(volumeCoordinate, VelocityVolumeY, markerY0, 1);
            float markerZ0 = texelFetch(MarkerVolume, volumeCoordinate - ivec3(0, 0, 1), 0).x;
            divergence += computeDivergenceForDirection(volumeCoordinate, VelocityVolumeZ, markerZ0, 2);
        }

        scale = clamp(divergence * 10.0 * Rendering.FluidGridToWorldScale, -1.0, 1.0);
        out_Tint = colormapCoolToWarm(scale);
        break;

    case VISUALIZE_PRESSURE_VELOCITY:
        float pressureV = marker == CELL_FLUID ? texelFetch(PressureVolume_Velocity, volumeCoordinate, 0).x : 0.0;
        scale = pressureV * Rendering.FluidGridToWorldScale;
        out_Tint = colormapCoolToWarm(pressureV).rgb;
        break;

    case VISUALIZE_PRESSURE_DENSITY:
        float pressureD = marker == CELL_FLUID ? texelFetch(PressureVolume_Density, volumeCoordinate, 0).x : 0.0;
        scale = pressureD * Rendering.FluidGridToWorldScale;
        out_Tint = colormapCoolToWarm(pressureD).rgb;
        break;

    case VISUALIZE_MARKER:
        scale = marker == CELL_AIR ? 0.0 : 1.0;

        if (marker == CELL_SOLID)
            out_Tint = vec3(0.0, 0.0, 0.0);
        else if (marker == CELL_FLUID)
            out_Tint = vec3(0.0, 0.0, 1.0);
        break;

#ifdef DEBUG
    case VISUALIZE_DEBUG:
        float debugValue = texelFetch(DebugVolume, volumeCoordinate, 0).x;
        scale = saturate(abs(debugValue));
        out_Tint = colormapCoolToWarm(debugValue).rgb;
        break;
#endif
    }
    scale = saturate(abs(scale));

    out_ParticleWorldPosition = (volumeCoordinate + vec3(0.5)) * Rendering.FluidGridToWorldScale + Rendering.FluidWorldMin;
    out_Radius = scale * 0.5 * Rendering.FluidGridToWorldScale;
    out_WorldPosition = spanParticle(out_ParticleWorldPosition, out_Radius);
    gl_Position = Camera.ViewProjection * vec4(out_WorldPosition, 1.0);
}
