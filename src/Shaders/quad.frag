#version 450
layout(location = 0) in vec2 fragUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform UBO {
    vec4 resolution;  // vec4 because of padding
};

void main() {
    vec2 uv = fragUV;
    uv.y = 1.0-uv.y;
    uv = uv*2.0-1.0;
    uv.x *= resolution.x/resolution.y;
    outColor = vec4(uv, 0.0, 1.0);
}
