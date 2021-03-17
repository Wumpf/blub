#version 460

#include "global_bindings.glsl"
#include "utilities.glsl"

layout(location = 0) in vec3 in_Position;
layout(push_constant) uniform PushConstants_ { uint MeshIndex; };

layout(location = 0) out vec3 out_VoxelCoordinates;
layout(location = 1) out uint out_SideIndex;

out gl_PerVertex { vec4 gl_Position; };

void main() {
    vec3 worldPosition = (vec4(in_Position, 1.0) * Meshes[MeshIndex].Transform).xyz;
    out_VoxelCoordinates = (worldPosition - Rendering.FluidWorldMin) / Rendering.FluidGridToWorldScale;

    // TODO: render from 3 sides (we have 3 instances for this purpose!)
    out_SideIndex = gl_InstanceIndex;
    float viewportSize = float(max3(Rendering.FluidGridResolution));

    vec3 swizzledVoxelCoordinates = out_VoxelCoordinates;
    if (gl_InstanceIndex == 0)
        swizzledVoxelCoordinates = swizzledVoxelCoordinates.zyx;
    if (gl_InstanceIndex == 1)
        swizzledVoxelCoordinates = swizzledVoxelCoordinates.xzy;

    gl_Position.xy = swizzledVoxelCoordinates.xy / viewportSize * 2.0 - 1.0;
    gl_Position.z = swizzledVoxelCoordinates.z / viewportSize;
    gl_Position.w = 1.0;
}
