layout(set = 1, binding = 0) uniform textureCube CubemapRgbe;
// todo: Constant buffer with a bunch of stuff.

vec3 decodeRGBE(vec4 hdr) { return hdr.rgb * exp2((hdr.a * 255.0) - 128.0); }

vec3 sampleHdrCubemap(vec3 dir) {
    vec4 rgbe = texture(samplerCube(CubemapRgbe, SamplerTrilinearClamp), dir);
    return decodeRGBE(rgbe);
}
