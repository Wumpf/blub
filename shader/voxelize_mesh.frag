#version 460

layout(push_constant) uniform PushConstants_ { uint MeshIndex; };

layout(set = 1, binding = 0, r8ui) uniform restrict coherent uimage3D VoxelVolume;

layout(location = 0) in vec3 in_VoxelCoordinates;
layout(location = 1) in flat uint in_SideIndex;

layout(location = 0) out float out_Dummy;

void main() {
    imageStore(VoxelVolume, ivec3(in_VoxelCoordinates), uvec4(1));
    out_Dummy = 0.0;
}