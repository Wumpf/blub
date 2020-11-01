#include "utilities.glsl"

// from: https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
float fresnelDielectricDielectric(float cosTheta, float eta) {
    float c = cosTheta;
    float temp = eta * eta + c * c - 1;

    if (temp < 0)
        return 1;

    float g = sqrt(temp);
    return 0.5 * sq((g - c) / (g + c)) * (1 + sq(((g + c) * c - 1) / ((g - c) * c + 1)));
}

float schlickFresnel(float nDotV, float R0) {
    float base = 1.0 - nDotV;
    float exponential = pow(base, 5.0);
    return exponential + R0 * (1.0 - exponential);
}

float evaluateNormalizedBlinnPhong(float BlinnPhongExponent, vec3 normal, vec3 toCamera, vec3 toLight) {
    vec3 halfVector = normalize(toCamera + toLight);
    float specularAmount = pow(saturate(dot(normal, halfVector)), BlinnPhongExponent);
    specularAmount *= (BlinnPhongExponent + 2.0) * INV_TAU; // Normalization factor

    return specularAmount;
}

// https://oceanopticsbook.info/view/scattering/the-henyey-greenstein-phase-function
// g is the average cosine between ingoing and outgoing light
// g==0 for isotropic scattering, 1 for max forward-scattering, -1 for max back-scattering
vec3 evaluateHenyeyGreensteinPhaseFunction(vec3 g, vec3 rayIn, vec3 rayOut) {
    const vec3 gSq = g * g;
    return (1.0f - gSq) * pow(1.0f + gSq - 2.0f * g * dot(rayIn, rayOut), vec3(-3.0f / 2.0f)) * (0.5 * INV_TAU);
}