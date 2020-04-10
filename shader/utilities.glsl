float saturate(float x) { return clamp(x, 0.0, 1.0); }

float lengthsq(vec3 a, vec3 b) {
    vec3 v = a - b;
    return dot(v, v);
}