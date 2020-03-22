#version 450

layout(location = 0) in vec2 in_QuadPosition;
layout(location = 0) out vec4 out_Color;

void main()
{
    const float radius = 1.0;
    float midDistSq = dot(in_QuadPosition, in_QuadPosition);
    if (midDistSq > radius) // todo: antialias
        discard;

    float height = sqrt(radius * radius - midDistSq);
    
    vec3 normalFlat = vec3(in_QuadPosition.x, height, in_QuadPosition.y);

    //float singleValueDisplay = length(normalFlat) > 1.0 ? 0.0 : 1.0;
    //out_Color = vec4(singleValueDisplay, singleValueDisplay, singleValueDisplay, 1.0);
    vec3 threeValueDisplay = normalFlat;
    out_Color = vec4(threeValueDisplay, 1.0);
}
