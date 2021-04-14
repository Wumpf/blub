#version 460

#include "../global_bindings.glsl"
#include "../utilities.glsl"

layout(push_constant) uniform PushConstants_ { uint MeshIndex; };

layout(location = 0) out flat uint out_SideIndex;

out gl_PerVertex { vec4 gl_Position; };

void main() {
    // Find out in which direction the current triangle pointing primarily.
    uint triangleBaseIdx = gl_VertexIndex / 3 * 3;
    vec3 trianglePosVoxel[3] = {
        (vec4(MeshVertices[MeshIndices[triangleBaseIdx + 0]].Position, 1.0) * Meshes[MeshIndex].VoxelTransform).xyz,
        (vec4(MeshVertices[MeshIndices[triangleBaseIdx + 1]].Position, 1.0) * Meshes[MeshIndex].VoxelTransform).xyz,
        (vec4(MeshVertices[MeshIndices[triangleBaseIdx + 2]].Position, 1.0) * Meshes[MeshIndex].VoxelTransform).xyz,
    };
    vec3 triangleNormalAbs = abs(cross(trianglePosVoxel[1] - trianglePosVoxel[0], trianglePosVoxel[2] - trianglePosVoxel[0]));
    out_SideIndex = triangleNormalAbs.x > triangleNormalAbs.y ? 0 : 1;
    out_SideIndex = triangleNormalAbs[out_SideIndex] > triangleNormalAbs.z ? out_SideIndex : 2;

    vec3 voxelCoordinates = trianglePosVoxel[gl_VertexIndex % 3];
    vec3 swizzledVoxelCoordinates;

    // Dominant X
    if (out_SideIndex == 0)
        swizzledVoxelCoordinates = voxelCoordinates.zyx;
    // Dominant Y
    else if (out_SideIndex == 1)
        swizzledVoxelCoordinates = voxelCoordinates.xzy;
    // Dominant Z
    else
        swizzledVoxelCoordinates = voxelCoordinates.xyz;

    float viewportSize = float(max3(Rendering.FluidGridResolution));
    gl_Position.x = swizzledVoxelCoordinates.x / viewportSize * 2.0 - 1.0;
    gl_Position.y = 1.0 - swizzledVoxelCoordinates.y / viewportSize * 2.0;
    gl_Position.z = swizzledVoxelCoordinates.z / viewportSize;
    gl_Position.w = 1.0;
}
