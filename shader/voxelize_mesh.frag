#version 460

#include "global_bindings.glsl"
#include "utilities.glsl"

layout(push_constant) uniform PushConstants_ { uint MeshIndex; };

layout(set = 1, binding = 0, r8ui) uniform restrict coherent uimage3D VoxelVolume;

layout(location = 0) in flat uint in_SideIndex;

layout(location = 0) out float out_Dummy;

ivec3 UnswizzlePos(ivec3 pos) { return in_SideIndex == 0 ? pos.zyx : (in_SideIndex == 1 ? pos.xzy : pos.xyz); }

void main() {
    // Retrieve voxel pos from gl_FragCoord
    // Careful: This voxel pos is still swizzled!
    float viewportSize = float(max3(Rendering.FluidGridResolution));
    vec3 voxelPosSwizzled;
    voxelPosSwizzled.xy = gl_FragCoord.xy;
    voxelPosSwizzled.z = gl_FragCoord.z * viewportSize;

    imageStore(VoxelVolume, UnswizzlePos(ivec3(voxelPosSwizzled)), uvec4(1));

    // "Depth Conservative"
    // If there is a strong change in depth we need to mark extra more voxels
    float depthDx = dFdxCoarse(voxelPosSwizzled.z);
    float depthDy = dFdyCoarse(voxelPosSwizzled.z);
    // Version I used in the path. I'm not entirely sure how I arrived that that...
    // float maxChange = length(vec2(depthDx, depthDy)) * 1.414; // * inversesqrt(2);
    // This makes more sense to me now and generates less clutter voxels
    float maxChange = max(abs(depthDx), abs(depthDy));

    if (floor(voxelPosSwizzled.z) != floor(voxelPosSwizzled.z - maxChange)) {
        imageStore(VoxelVolume, UnswizzlePos(ivec3(voxelPosSwizzled - vec3(0, 0, 1))), uvec4(1));
    }
    if (floor(voxelPosSwizzled.z) != floor(voxelPosSwizzled.z + maxChange)) {
        imageStore(VoxelVolume, UnswizzlePos(ivec3(voxelPosSwizzled + vec3(0, 0, 1))), uvec4(1));
    }

    out_Dummy = 0.0;
}