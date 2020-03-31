layout(binding = 0, set = 0) uniform Camera {
    mat4 ViewProjection;
    vec3 CameraPosition;
    vec3 CameraRight;
    vec3 CameraUp;
    vec3 CameraDirection;
};
