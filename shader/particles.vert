#version 450

out gl_PerVertex {
    vec4 gl_Position;
};

layout(location = 0) out vec2 out_QuadPosition;

layout(binding = 0, set = 0) uniform Camera {
    mat4 ViewProjection;
    vec3 CameraPosition;
    vec3 CameraRight;
    vec3 CameraUp;
    vec3 CameraDirection;
};

const vec2 quadPositions[4] = vec2[4](
    vec2(-1.0, -1.0),
    vec2(-1.0, 1.0),
    vec2(1.0, -1.0),
    vec2(1.0, 1.0)
);

void main() {
    const float radius = 0.45;
    out_QuadPosition = quadPositions[gl_VertexIndex];

    vec2 pos2d = quadPositions[gl_VertexIndex] * radius;

    vec3 particleWorldPosition = vec3(gl_InstanceIndex % 10, gl_InstanceIndex / 10, 0.0); // this comes from a buffer later
    vec3 particleNormal = normalize(CameraPosition - particleWorldPosition);
    vec3 particleRight = cross(particleNormal, CameraUp);   // Not sure about the orientation of any of those.
    vec3 particleUp = cross(particleRight, particleNormal);

    vec3 worldPos = particleWorldPosition + pos2d.x * particleRight + pos2d.y * particleUp;

    gl_Position = ViewProjection * vec4(worldPos, 1.0);
}
