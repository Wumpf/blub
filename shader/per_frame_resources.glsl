struct CameraData {
    mat4 ViewProjection;
    vec3 Position;
    vec3 Right;
    vec3 Up;
    vec4 Direction;
};

layout(set = 0, binding = 0) uniform PerFrameConstants {
    CameraData Camera;

    float TotalPassedTime;
    float DeltaTime; // How long a previous frame took in seconds
};

layout(set = 0, binding = 1) uniform sampler SamplerTrilinear;