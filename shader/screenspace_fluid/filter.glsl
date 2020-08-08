layout(push_constant) uniform PushConstants { uint FilterDirection; };

#if defined(FILTER_2D)

uvec2 getScreenCoord() { return gl_GlobalInvocationID.xy; }
uvec2 getBlockScreenCoord() { return gl_WorkGroupID.xy * gl_WorkGroupSize.xy; }

#else

uvec2 getScreenCoord() {
    if (FilterDirection == 1)
        return gl_WorkGroupID.xy * gl_WorkGroupSize.yx + gl_LocalInvocationID.yx;
    else
        return gl_GlobalInvocationID.xy;
}

uvec2 getBlockScreenCoord() {
    if (FilterDirection == 1)
        return gl_WorkGroupID.xy * gl_WorkGroupSize.yx;
    else
        return gl_WorkGroupID.xy * gl_WorkGroupSize.xy;
}

#endif