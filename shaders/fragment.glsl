#version 330
in float v_height;
in vec2 v_uv;
in vec3 v_pos;
out vec4 f_color;
uniform float WORLD_L;
uniform sampler2D height_map;
uniform vec3 light_vector;
uniform vec3 cam_pos;
void main() {
    float texel = 1.0 / 256.0;
    float h_l = texture(height_map, v_uv + vec2(-texel, 0.0)).r;
    float h_r = texture(height_map, v_uv + vec2( texel, 0.0)).r;
    float h_d = texture(height_map, v_uv + vec2(0.0, -texel)).r;
    float h_u = texture(height_map, v_uv + vec2(0.0,  texel)).r;
    vec2 grad = vec2(h_r-h_l, h_u-h_d) / (2.0*texel*WORLD_L);
    vec3 N = normalize(vec3(-grad.x, 1.0, -grad.y));
    vec3 V = normalize(cam_pos - v_pos);
    vec3 H = normalize(light_vector + V);
    float spec = pow(max(dot(N,H),0.0), 128.0);
    vec3 base_color = vec3(0.04, 0.18, 0.38);
    float diffuse = pow(max(dot(N,light_vector),0.0), 2.0);
    float F0 = 0.02;
    float fresnel = F0 + (1.0-F0)*pow(1.0-max(dot(N,V),0.0),5.0);
    fresnel = clamp(fresnel,0.0,1.0);
    vec3 color = base_color*(1.0-fresnel) + vec3(0.8,0.9,1.0)*fresnel;
    color += spec * vec3(1.0, 0.95, 0.85) * 0.6;
    f_color = vec4(color, 1.0);
}