#ifndef INCLUDE_PERFRAMERESOURCES
#define INCLUDE_PERFRAMERESOURCES

struct CameraData {
    mat4 ViewProjection;
    vec3 Position;
    vec3 Right;
    vec3 Up;
    vec4 Direction;
};

// All timings in seconds.
struct TimerData {
    float TotalPassed;        // How much time has passed on the rendering clock since rendering started (including the current frame).
    float FrameDelta;         // How long a previous frame took in seconds.
    float TotalSimulatedTime; // How much time has passed in the simulation *excluding any steps in the current frame*.
    // How much we're advancing the simulation for each step in the current frame.
    // (This implicitly assumes that all steps are equally sized during a frame which may become invalid if we ever attempt a fully adaptive step!)
    float SimulationDelta;
};

struct GlobalRenderingSettings {
    vec3 FluidWorldOrigin;
    float FluidGridToWorldScale;
    float VelocityVisualizationScale;
};

// Constants that change at max per frame.
// (might group a few even more constant data into here as well - a few bytes updated more or less won't make a difference in render time!)
layout(set = 0, binding = 0) uniform PerFrameConstants {
    CameraData Camera;
    TimerData Time;
    GlobalRenderingSettings Rendering;
};

layout(set = 0, binding = 1) uniform sampler SamplerTrilinearClamp;
layout(set = 0, binding = 2) uniform sampler SamplerPointClamp;

// See HdrBackbuffer::FORMAT
#define HDR_BACKBUFFER_IMAGE_FORMAT rgba16f

#endif // INCLUDE_PERFRAMERESOURCES