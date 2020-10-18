#version 450

out gl_PerVertex { vec4 gl_Position; };

const vec2 triPositions[3] = vec2[3](vec2(-1.0, -3.0), vec2(-1.0, 1.0), vec2(3.0, 1.0));

void main() { gl_Position = vec4(triPositions[gl_VertexIndex], 1.0, 1.0); }
