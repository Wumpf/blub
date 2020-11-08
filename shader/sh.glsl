// Evalutes SH bands 0,1,2 for a given direction

#define SH_FACTOR_BAND0 0.282094792 // 1.0 / (2.0 * sqrt(PI))

#define SH_FACTOR_BAND1 0.488602512      // sqrt(3.0) / (2.0 * sqrt(PI))
#define SH_FACTOR_BAND2_non0 1.092548431 // sqrt(15.0) / (2.0 * sqrt(PI))
#define SH_FACTOR_BAND2_0 0.315391565    // sqrt(5.0) / (4.0 * sqrt(PI))

#define SH_FACTOR_BAND3_0 0.373176336
#define SH_FACTOR_BAND3_1 0.457045794
#define SH_FACTOR_BAND3_2 2.89061141
#define SH_FACTOR_BAND3_3 0.590043604

#define SH_FACTOR_COSINE_BAND0 0.886226925 // PI / (2.0 * sqrt(PI))

#define SH_FACTOR_COSINE_BAND1 1.023326708      // 2.0 * PI * sqrt(3.0) / (6.0 * sqrt(PI))
#define SH_FACTOR_COSINE_BAND2_non0 0.858085531 // PI * sqrt(15.0) / (8.0 * sqrt(PI))
#define SH_FACTOR_COSINE_BAND2_0 0.247707956    // PI * sqrt(5.0) / (16.0 * sqrt(PI))

vec3 sh3Evaluate(vec3 dir, vec3 shCoeffs[9]) {
    vec3 result = shCoeffs[0] * SH_FACTOR_BAND0;
    result += shCoeffs[1] * (-SH_FACTOR_BAND1 * dir.y);
    result += shCoeffs[2] * (SH_FACTOR_BAND1 * dir.z);
    result += shCoeffs[3] * (-SH_FACTOR_BAND1 * dir.x);
    result += shCoeffs[4] * (SH_FACTOR_BAND2_non0 * dir.y * dir.x);
    result += shCoeffs[5] * (-SH_FACTOR_BAND2_non0 * dir.y * dir.z);
    result += shCoeffs[6] * (SH_FACTOR_BAND2_0 * (3.0 * dir.z * dir.z - 1.0));
    result += shCoeffs[7] * (-SH_FACTOR_BAND2_non0 * dir.x * dir.z);
    result += shCoeffs[8] * (SH_FACTOR_BAND2_non0 * 0.5 * (dir.x * dir.x - dir.y * dir.y));
    return max(vec3(0.0), result);
}

vec3 sh3EvaluateCosine(vec3 dir, vec3 shCoeffs[9]) {
    vec3 result = shCoeffs[0] * SH_FACTOR_COSINE_BAND0;
    result += shCoeffs[1] * (-SH_FACTOR_COSINE_BAND1 * dir.y);
    result += shCoeffs[2] * (SH_FACTOR_COSINE_BAND1 * dir.z);
    result += shCoeffs[3] * (-SH_FACTOR_COSINE_BAND1 * dir.x);
    result += shCoeffs[4] * (SH_FACTOR_COSINE_BAND2_non0 * dir.y * dir.x);
    result += shCoeffs[5] * (-SH_FACTOR_COSINE_BAND2_non0 * dir.y * dir.z);
    result += shCoeffs[6] * (SH_FACTOR_COSINE_BAND2_0 * (3.0 * dir.z * dir.z - 1.0));
    result += shCoeffs[7] * (-SH_FACTOR_COSINE_BAND2_non0 * dir.x * dir.z);
    result += shCoeffs[8] * (SH_FACTOR_COSINE_BAND2_non0 * 0.5 * (dir.x * dir.x - dir.y * dir.y));
    return max(vec3(0.0), result);
}

vec3 sh4Evaluate(vec3 dir, vec3 shCoeffs[16]) {
    vec3 result = shCoeffs[0] * SH_FACTOR_BAND0;

    result += shCoeffs[1] * (-SH_FACTOR_BAND1 * dir.y);
    result += shCoeffs[2] * (SH_FACTOR_BAND1 * dir.z);
    result += shCoeffs[3] * (-SH_FACTOR_BAND1 * dir.x);

    result += shCoeffs[4] * (SH_FACTOR_BAND2_non0 * dir.y * dir.x);
    result += shCoeffs[5] * (-SH_FACTOR_BAND2_non0 * dir.y * dir.z);
    result += shCoeffs[6] * (SH_FACTOR_BAND2_0 * (3.0 * dir.z * dir.z - 1.0));
    result += shCoeffs[7] * (-SH_FACTOR_BAND2_non0 * dir.x * dir.z);
    result += shCoeffs[8] * (SH_FACTOR_BAND2_non0 * 0.5 * (dir.x * dir.x - dir.y * dir.y));

    result += shCoeffs[9] * (-SH_FACTOR_BAND3_3 * dir.y * (3.0 * dir.x * dir.x - dir.y * dir.y));
    result += shCoeffs[10] * (SH_FACTOR_BAND3_2 * dir.x * dir.y * dir.z);
    result += shCoeffs[11] * (-SH_FACTOR_BAND3_1 * dir.y * (5.0 * dir.z * dir.z - 1.0));
    result += shCoeffs[12] * (SH_FACTOR_BAND3_0 * dir.z * (5.0 * dir.z * dir.z - 3.0));
    result += shCoeffs[13] * (-SH_FACTOR_BAND3_1 * dir.x * (5.0 * dir.z * dir.z - 1.0));
    result += shCoeffs[14] * (SH_FACTOR_BAND3_2 * (0.5 * (dir.x * dir.x - dir.y * dir.y) * dir.z));
    result += shCoeffs[15] * (-SH_FACTOR_BAND3_3 * dir.x * (dir.x * dir.x - 3.0 * dir.y * dir.y));

    return max(vec3(0.0), result);
}
// Wasn't able to confirm myself, but several sources state that the (zonal) coefficients for
// Henyey Greenstein are simply just g^n (n being the band index)
// See:
// * http://sjbrown.co.uk/2004/10/16/spherical-harmonic-basis/
// * https://bartwronski.files.wordpress.com/2014/08/bwronski_volumetric_fog_siggraph2014.pdf
// * https://books.google.se/books?id=dDL3DwAAQBAJ&pg=PA325&lpg=PA325&dq=henyey+greenstein+zonal+harmonics&source=bl&ots=uN8TFxkSLx&sig=ACfU3U2wZ-poPmmigPUxpDuOaUrchcLxxA&hl=en&sa=X&ved=2ahUKEwjPtLW4yM3sAhXososKHVVeD7kQ6AEwCXoECAgQAg#v=onepage&q=henyey%20greenstein%20zonal%20harmonics&f=false
vec3 sh3EvaluateHenyeyGreensteinPhaseFunction(vec3 g, vec3 dir, vec3 shCoeffs[9]) {
    vec3 gSq = sq(g);
    dir = -dir;

    vec3 result = shCoeffs[0] * SH_FACTOR_BAND0;
    result += g * shCoeffs[1] * (-SH_FACTOR_BAND1 * dir.y);
    result += g * shCoeffs[2] * (SH_FACTOR_BAND1 * dir.z);
    result += g * shCoeffs[3] * (-SH_FACTOR_BAND1 * dir.x);
    result += gSq * shCoeffs[4] * (SH_FACTOR_BAND2_non0 * dir.y * dir.x);
    result += gSq * shCoeffs[5] * (-SH_FACTOR_BAND2_non0 * dir.y * dir.z);
    result += gSq * shCoeffs[6] * (SH_FACTOR_BAND2_0 * (3.0 * dir.z * dir.z - 1.0));
    result += gSq * shCoeffs[7] * (-SH_FACTOR_BAND2_non0 * dir.x * dir.z);
    result += gSq * shCoeffs[8] * (SH_FACTOR_BAND2_non0 * 0.5 * (dir.x * dir.x - dir.y * dir.y));
    return max(vec3(0.0), result);
}

vec3 sh4EvaluateHenyeyGreensteinPhaseFunction(vec3 g, vec3 dir, vec3 shCoeffs[16]) {
    vec3 gSq = sq(g);
    vec3 ggg = gSq * g;
    dir = -dir;

    vec3 result = shCoeffs[0] * SH_FACTOR_BAND0;
    result += g * shCoeffs[1] * (-SH_FACTOR_BAND1 * dir.y);
    result += g * shCoeffs[2] * (SH_FACTOR_BAND1 * dir.z);
    result += g * shCoeffs[3] * (-SH_FACTOR_BAND1 * dir.x);
    result += gSq * shCoeffs[4] * (SH_FACTOR_BAND2_non0 * dir.y * dir.x);
    result += gSq * shCoeffs[5] * (-SH_FACTOR_BAND2_non0 * dir.y * dir.z);
    result += gSq * shCoeffs[6] * (SH_FACTOR_BAND2_0 * (3.0 * dir.z * dir.z - 1.0));
    result += gSq * shCoeffs[7] * (-SH_FACTOR_BAND2_non0 * dir.x * dir.z);
    result += gSq * shCoeffs[8] * (SH_FACTOR_BAND2_non0 * 0.5 * (dir.x * dir.x - dir.y * dir.y));
    result += ggg * shCoeffs[9] * (-SH_FACTOR_BAND3_3 * dir.y * (3.0 * dir.x * dir.x - dir.y * dir.y));
    result += ggg * shCoeffs[10] * (SH_FACTOR_BAND3_2 * dir.x * dir.y * dir.z);
    result += ggg * shCoeffs[11] * (-SH_FACTOR_BAND3_1 * dir.y * (5.0 * dir.z * dir.z - 1.0));
    result += ggg * shCoeffs[12] * (SH_FACTOR_BAND3_0 * dir.z * (5.0 * dir.z * dir.z - 3.0));
    result += ggg * shCoeffs[13] * (-SH_FACTOR_BAND3_1 * dir.x * (5.0 * dir.z * dir.z - 1.0));
    result += ggg * shCoeffs[14] * (SH_FACTOR_BAND3_2 * (0.5 * (dir.x * dir.x - dir.y * dir.y) * dir.z));
    result += ggg * shCoeffs[15] * (-SH_FACTOR_BAND3_3 * dir.x * (dir.x * dir.x - 3.0 * dir.y * dir.y));

    return max(vec3(0.0), result);
}