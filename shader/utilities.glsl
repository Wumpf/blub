#ifndef INCLUDE_UTILITIES
#define INCLUDE_UTILITIES

// Adds [[flatten]] and friends (which are understood by SPIR-V)
// https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GL_EXT_control_flow_attributes.txt
#extension GL_EXT_control_flow_attributes : require

// Workaround for bug in shaderc when using textureSize/texelFetch on textures (instead of samplers)
// error: 'textureSize' : required extension not requested: GL_EXT_samplerless_texture_functions
// Shouldn't happen since we compile for Vulkan.
#extension GL_EXT_samplerless_texture_functions : require

float saturate(float x) { return clamp(x, 0.0, 1.0); }
vec2 saturate(vec2 x) { return clamp(x, vec2(0.0), vec2(1.0)); }
vec3 saturate(vec3 x) { return clamp(x, vec3(0.0), vec3(1.0)); }
vec4 saturate(vec4 x) { return clamp(x, vec4(0.0), vec4(1.0)); }

float lengthsq(vec3 a, vec3 b) {
    vec3 v = a - b;
    return dot(v, v);
}

float sq(float a) { return a * a; }
vec2 sq(vec2 a) { return a * a; }
vec3 sq(vec3 a) { return a * a; }
vec4 sq(vec4 a) { return a * a; }

// t = [0; 1]
vec3 colormapHeat(float t) { return saturate(vec3(t * 3, t * 3 - 1, t * 3 - 2)); }
// t = [-1; 1]
vec3 colormapCoolToWarm(float t) { return t < 0.0 ? mix(vec3(1.0), vec3(0.0, 0.0, 1.0), -t) : mix(vec3(1.0), vec3(1.0, 0.0, 0.0), t); }

bool sphereIntersect(vec3 spherePosition, float radius, vec3 rayOrigin, vec3 rayDir, out float sphereDistance, out float intersectFar) {
    // Sphere intersect raycast.
    // (uses equation based intersect: rayOrigin + t * rayDir, ||sphereOrigin-pointOnSphere||= r*r, [...])
    vec3 particleCenterToCamera = rayOrigin - spherePosition; // (often denoted as oc == OriginCenter)
    float b = dot(particleCenterToCamera, rayDir);
    float c = dot(particleCenterToCamera, particleCenterToCamera) - radius * radius;
    float discriminant = b * b - c;
    if (discriminant < 0.0)
        return false;
    discriminant = sqrt(discriminant);
    sphereDistance = -b - discriminant;
    intersectFar = -b + discriminant;
    return true;
}

bool sphereIntersect(vec3 spherePosition, float radius, vec3 rayOrigin, vec3 rayDir, out float sphereDistance) {
    float intersectFar;
    return sphereIntersect(spherePosition, radius, rayOrigin, rayDir, sphereDistance, intersectFar);
}

float dot2(in vec3 v) { return dot(v, v); }
float max3(vec3 v) { return max(max(v.x, v.y), v.z); }
float max3(ivec3 v) { return max(max(v.x, v.y), v.z); }
float max3(uvec3 v) { return max(max(v.x, v.y), v.z); }
float min3(vec3 v) { return min(min(v.x, v.y), v.z); }

// https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
// box is centered at 0 and has extents of b.
// float sdBox(vec3 p, vec3 b) {
//     vec3 q = abs(p) - b;
//     return length(max(q, 0.0)) + min(max3(q), 0.0);
// }

// also known as 1/(2pi)
#define INV_TAU 0.15915494309
#define PI 3.14159265359

#endif // INCLUDE_UTILITIES