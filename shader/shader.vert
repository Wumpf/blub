#version 450

out gl_PerVertex {
    vec4 gl_Position;
};

layout(binding = 0, set = 0) uniform Camera {
    mat4 ViewProjection;
};

const vec2 positions[3] = vec2[3](
    vec2(0.0, -0.5),
    vec2(0.5, 0.5),
    vec2(-0.5, 0.5)
);

void main() {
    gl_Position = ViewProjection * vec4(positions[gl_VertexIndex], 0.0, 1.0);
}
