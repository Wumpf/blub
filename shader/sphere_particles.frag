#version 450

layout(location = 0) in vec3 in_WorldPosition;
layout(location = 1) in vec3 in_ParticleWorldPosition;
layout(location = 0) out vec4 out_Color;

// todo: includes
layout(binding = 0, set = 0) uniform Camera {
    mat4 ViewProjection;
    vec3 CameraPosition;
    vec3 CameraRight;
    vec3 CameraUp;
    vec3 CameraDirection;
};
float saturate(float x)
{
    return clamp(x, 0.0, 1.0);
}

void main()
{
    const float radius = 0.5; // todo.
    const vec3 lightdir = normalize(vec3(1.0, 2.0, 1.0));

    // Sphere intersect raycast. Given how obscure our vertex positions are, this is the easiest!
    // (uses equation based intersect: rayOrigin + t * rayDir, ||sphereOrigin-pointOnSphere||= r*r, [...])
    vec3 rayDir = normalize(in_WorldPosition - CameraPosition);
    vec3 particleCenterToCamera = CameraPosition - in_ParticleWorldPosition; // (often denoted as oc == OriginCenter)
    float b = dot(particleCenterToCamera, rayDir);
    float c = dot(particleCenterToCamera, particleCenterToCamera) - radius * radius;
    float discr = b*b - c;
    if (discr < 0.0)
        discard; // todo: antialias?
    float cameraDistance = -b - sqrt(discr);

    vec3 sphereWorldPos = CameraPosition + cameraDistance * rayDir;
    vec3 normal = (sphereWorldPos - in_ParticleWorldPosition) / radius;
    
    // via https://www.iquilezles.org/www/articles/outdoorslighting/outdoorslighting.htm
    float sun = saturate(dot(lightdir, normal));
    float sky = saturate(0.5 + 0.5*normal.y);
    float ind = saturate(dot(normal, normalize(lightdir * vec3(-1.0, 0.0, -1.0))));
    vec3 lighting = sun * vec3(1.64,1.27,0.99);
    lighting += sky * vec3(0.16,0.20,0.28);
    lighting += ind * vec3(0.40,0.28,0.20);

    out_Color = vec4(lighting, 1.0);
}
