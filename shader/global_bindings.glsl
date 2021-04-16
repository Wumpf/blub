#ifndef INCLUDE_PERFRAMERESOURCES
#define INCLUDE_PERFRAMERESOURCES

// ----------------------------------------
// Constants
// ----------------------------------------

struct CameraData {
    mat4 ViewProjection;
    vec3 Position;
    vec3 Right;
    vec3 Up;
    vec3 Direction;

    // NDC (1, 1) projected into camera space and then divded by the far plane distance.
    vec2 NdcCameraSpaceProjected;
    float TanHalfVerticalFov;    // tan(VerticalFov * 0.5)
    float InvTanHalfVerticalFov; // 1.0 / TanHalfVerticalFov
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
    vec3 FluidWorldMin;          // Origin of the fluid domain in world space
    float FluidGridToWorldScale; // how big is a grid cell in world scale
    vec3 FluidWorldMax;
    float VelocityVisualizationScale;
    uvec3 FluidGridResolution; // TODO: This is not a rendering setting
    float FluidParticleRadius; // particle size in world space
};

struct ScreenData {
    vec2 Resolution;
    vec2 ResolutionInv; // 1.0 / Resolution
};

// Constants that change at max per frame.
// (might group a few even more constant data into here as well - a few bytes updated more or less won't make a difference in render time!)
layout(set = 0, binding = 0) uniform PerFrameConstants {
    CameraData Camera;
    TimerData Time;
    GlobalRenderingSettings Rendering;
    ScreenData Screen;
};

// ----------------------------------------
// Sampler
// ----------------------------------------

layout(set = 0, binding = 1) uniform sampler SamplerTrilinearClamp;
layout(set = 0, binding = 2) uniform sampler SamplerPointClamp;

// ----------------------------------------
// Mesh data
// ----------------------------------------

struct MeshData {
    // transposed so we can store them in 3 vec4s.
    mat3x4 WorldTransform; // Transforms from object space to world space
    mat3x4 VoxelTransform; // Transforms from object space to voxel space (equals world space but with scaling and extra translation)
    vec3 FluidSpaceVelocity;
    vec3 FluidSpaceRotationAxisScaled; // Rotation axis scaled with angular velocity (in radians)

    uvec2 VertexBufferRange;
    uvec2 IndexBufferRange;

    int TextureIndex;
    // ivec3 _Padding1;
};
layout(set = 0, binding = 3) restrict readonly buffer Meshes_ { MeshData Meshes[]; };
// Not going with dynamic size (UNSIZED_BINDING_ARRAY extension) for convenience in layout setup (which doesn't change per scene).
// (also this is more widely supported)
layout(set = 0, binding = 4) uniform texture2D MeshTextures[1];

// Can't do packed layouts in glsl/spirv?
struct Vertex {
    vec3 Position;
    float NormalX;
    vec2 NormalYZ;
    vec2 Texcoord;
};
layout(set = 0, binding = 5) restrict readonly buffer MeshIndices_ { uint MeshIndices[]; };
layout(set = 0, binding = 6) restrict readonly buffer MeshVertices_ { Vertex MeshVertices[]; };

// ----------------------------------------
// Other
// ----------------------------------------

// See HdrBackbuffer::FORMAT
#define HDR_BACKBUFFER_IMAGE_FORMAT rgba16f

// Computes world space position from standard depth buffer depth.
// (using "classic depth buffer", as defined in with our global camera matrices)
vec3 reconstructWorldPositionFromViewSpaceDepth(vec2 screenUv, float depth) {
    float x = screenUv.x * 2.0f - 1.0f;
    float y = (1.0 - screenUv.y) * 2.0f - 1.0f;
    vec3 viewSpace = vec3(Camera.NdcCameraSpaceProjected * vec2(x, y) * depth, depth);
    return viewSpace.x * Camera.Right + viewSpace.y * Camera.Up + viewSpace.z * Camera.Direction + Camera.Position;
}

#endif // INCLUDE_PERFRAMERESOURCES