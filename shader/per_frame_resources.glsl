struct CameraData {
    mat4 ViewProjection;
    vec3 Position;
    vec3 Right;
    vec3 Up;
    vec4 Direction;
};

// All timings in seconds.
struct TimerData {
    float TotalPassed;           // How much time has passed in the real world since rendering started.
    float FrameDelta;            // How long a previous frame took in seconds
    float SimulationTotalPassed; // How much time has passed in the simulation *excluding any steps in the current frame*.
    // How much we're advancing the simulation for each step in the current frame.
    // (This implicitly assumes that all steps are equally sized during a frame which may become invalid if we ever attempt a fully adaptive step!)
    float SimulationDelta;
};

// Constants that change at max per frame.
// (might group a few even more constant data into here as well - a few bytes updated more or less won't make a difference in render time!)
layout(set = 0, binding = 0) uniform PerFrameConstants {
    CameraData Camera;
    TimerData Time;
};

layout(set = 0, binding = 1) uniform sampler SamplerTrilinearClamp;
layout(set = 0, binding = 2) uniform sampler SamplerPointClamp;