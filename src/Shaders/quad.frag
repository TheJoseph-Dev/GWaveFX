#version 450
layout(location = 0) in vec2 fragUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform UBO {
    vec4 resolution;  // vec4 because of padding
};

layout(set = 0, binding = 1) uniform sampler1D wSample;
layout(set = 0, binding = 2) uniform sampler1D wFreq;

void main() {
    vec2 uv = fragUV;
    uv.y = 1.0-uv.y;
    uv = uv*2.0-1.0;
    uv.x *= resolution.x/resolution.y;
    float sValue = texture(wSample, fragUV.x).r;
    float fValue = texture(wFreq, fragUV.x).r;
    if(uv.y > 0) outColor = vec4(sValue);
    else outColor = vec4(fValue);
}
