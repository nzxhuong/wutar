#version 330
in float v_height;
in vec2 v_uv;
in vec3 v_pos;
out vec4 f_color;
uniform float WORLD_L;
uniform sampler2D height_map;
uniform vec3 light_vector;
uniform vec3 cam_pos;
uniform samplerCube skybox;
uniform sampler2D foam_map; 
void main() {
    float texel = 1.0 / 1024.0;
    float h_l = texture(height_map, v_uv + vec2(-texel, 0.0)).r;
    float h_r = texture(height_map, v_uv + vec2( texel, 0.0)).r;
    float h_d = texture(height_map, v_uv + vec2(0.0, -texel)).r;
    float h_u = texture(height_map, v_uv + vec2(0.0,  texel)).r;
    vec2 grad = vec2(h_r-h_l, h_u-h_d) / (2.0*texel*WORLD_L);
    vec3 N = normalize(vec3(-grad.x, 1.0, -grad.y));
    vec3 V = normalize(cam_pos - v_pos);
    vec3 H = normalize(light_vector + V);
    float spec = pow(max(dot(N,H),0.0), 50.0);
    vec3 light_color = vec3(1.0, 1.0, 1.0);
    vec3 ambient_color = vec3(0.05, 0.13, 0.19);
    vec3 bubble_color = vec3(0.347, 0.513, 0.688);
    float fresnel = pow(1.0-max(dot(N,V),0.0),5.0);
    vec3 water_scatter_color = vec3(0.558, 0.754, 0.718);
    vec3 ambient = -0.1 * ambient_color  * light_color + 0.45 * bubble_color * light_color;
    float height = texture(height_map, v_uv).r;
    float water_scatter_1 = 0.51 * max(0.0, height) * pow(max(0.0, dot(light_vector, -V)), 4.0) * pow(max(0.0, 0.5 - 0.5 * dot(N, light_vector)), 3.0);
    float dotNV = dot(N, V);
    float water_scatter_2 = -0.10 * (dotNV * dotNV);
    vec3 water_scatter = water_scatter_color * light_color * (1.0 - fresnel) * (water_scatter_1 + water_scatter_2);
    vec3 reflection_color = texture(skybox, reflect(-V, N)).rgb * fresnel * 0.41;
    vec3 final_color = ambient + water_scatter + reflection_color + spec * vec3(3.688, 2.656, 1.931) * fresnel;
    float foam = texture(foam_map, v_uv).r;
    vec3 foam_color = vec3(1.0, 1.0, 1.0);
    float foam_blend = smoothstep(0.0, 0.65, foam);
    final_color = mix(final_color, foam_color, foam_blend) * max(light_color, vec3(0.4));

    f_color = vec4(final_color, 1.0);
}