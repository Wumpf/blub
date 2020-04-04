layout(set = 0, binding = 0) uniform Camera {
    mat4 ViewProjection;
    vec3 CameraPosition;
    vec3 CameraRight;
    vec3 CameraUp;
    vec3 CameraDirection;
};

layout(set = 0, binding = 1) uniform sampler SamplerTrilinear;