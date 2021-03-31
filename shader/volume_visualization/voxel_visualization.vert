
// https://twitter.com/donzanoid/status/616370134278606848?lang=en
/*

b = 1 << i;
x = (0x287a & b) != 0
y = (0x02af & b) != 0
z = (0x31e3 & b) != 0
*/

#version 460

#include "utilities.glsl"
#include "volume_visualization.glsl"

layout(set = 2, binding = 0) uniform texture3D SceneVoxelization;
layout(location = 0) out vec3 out_WorldPosition;
layout(location = 1) out flat ivec3 out_VolumeCoordinate;

vec3 getCubeCoordinate(uint vertexIndex) {
    // Idea courtesy of Don Williamson
    // https://twitter.com/donzanoid/status/616370134278606848?lang=en
    uint b = 1 << vertexIndex;
    return vec3((0x287a & b) != 0, (0x02af & b) != 0, (0x31e3 & b) != 0);
}

void main() {
    out_VolumeCoordinate = getVolumeCoordinate(gl_InstanceIndex);
    float voxelValue = texelFetch(SceneVoxelization, out_VolumeCoordinate, 0).w;
    if (voxelValue == 0) {
        gl_Position = vec4(-999999999.0);
        return;
    }

    vec3 cubeCoordinate = getCubeCoordinate(gl_VertexIndex);

    out_WorldPosition = (vec3(out_VolumeCoordinate) + cubeCoordinate) * Rendering.FluidGridToWorldScale + Rendering.FluidWorldMin;

    gl_Position = Camera.ViewProjection * vec4(out_WorldPosition, 1.0);
}