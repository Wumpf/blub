#version 460

#include "../global_bindings.glsl"
#include "../utilities.glsl"

layout(push_constant) uniform PushConstants_ { uint MeshIndex; };

layout(set = 1, binding = 0, rgba16f) uniform restrict writeonly image3D SceneVoxelization;

layout(location = 0) in flat uint in_SideIndex;

layout(location = 0) out float out_Dummy;

vec3 Unswizzle(vec3 v) { return in_SideIndex == 0 ? v.zyx : (in_SideIndex == 1 ? v.xzy : v.xyz); }
vec3 UnswizzlePosAndClamp(vec3 pos) { return clamp(Unswizzle(pos), vec3(0), vec3(Rendering.FluidGridResolution) - vec3(1)); }

vec3 ComputeVoxelSpeed(vec3 voxelPos) {
    vec3 voxelSpaceCenter = (vec4(0.0, 0.0, 0.0, 1.0) * Meshes[MeshIndex].VoxelTransform).xyz;
    vec3 a = Meshes[MeshIndex].FluidSpaceRotationAxisScaled;
    vec3 p = voxelPos - voxelSpaceCenter;
    vec3 tangentialVelocity = cross(a, p - dot(p, a) * a);
    return tangentialVelocity + Meshes[MeshIndex].FluidSpaceVelocity;
}

void main() {
    // Retrieve voxel pos from gl_FragCoord
    // Careful: This voxel pos is still swizzled!
    float viewportSize = float(max3(Rendering.FluidGridResolution));
    vec3 voxelPosSwizzled;
    voxelPosSwizzled.xy = gl_FragCoord.xy;
    voxelPosSwizzled.z = gl_FragCoord.z * viewportSize;
    vec3 voxelPos = UnswizzlePosAndClamp(ivec3(voxelPosSwizzled));
    imageStore(SceneVoxelization, ivec3(voxelPos), vec4(ComputeVoxelSpeed(voxelPos), 1.0));

    // "Depth Conservative"
    // If there is a strong change in depth we need to mark extra more voxels
    float depthDx = dFdxCoarse(voxelPosSwizzled.z);
    float depthDy = dFdyCoarse(voxelPosSwizzled.z);
    // Version I used in the past. I'm not entirely sure how I arrived that that...
    // float maxChange = length(vec2(depthDx, depthDy)) * 1.414; // * inversesqrt(2);
    // This makes more sense to me now and generates less clutter voxels
    float maxChange = max(abs(depthDx), abs(depthDy));

    if (floor(voxelPosSwizzled.z) != floor(voxelPosSwizzled.z - maxChange)) {
        voxelPos = UnswizzlePosAndClamp(voxelPosSwizzled - vec3(0, 0, 1));
        imageStore(SceneVoxelization, ivec3(voxelPos), vec4(ComputeVoxelSpeed(voxelPos), 1.0));
    }
    if (floor(voxelPosSwizzled.z) != floor(voxelPosSwizzled.z + maxChange)) {
        voxelPos = UnswizzlePosAndClamp(voxelPosSwizzled + vec3(0, 0, 1));
        imageStore(SceneVoxelization, ivec3(voxelPos), vec4(ComputeVoxelSpeed(voxelPos), 1.0));
    }

    out_Dummy = 0.0;
}