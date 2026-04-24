#version 330
in float v_height;
in vec3 v_normal;
in vec3 v_pos;
out vec4 f_color;
uniform vec3 light_vector;
uniform vec3 cam_pos;
void main() {
    vec3 N = normalize(v_normal);
    vec3 V = normalize(cam_pos - v_pos);
    vec3 H = normalize(light_vector + V);
    float spec = pow(max(dot(N, H), 0.0), 128.0);
    vec3 base_color = vec3(0.04, 0.18, 0.38);
    float diffuse = pow(max(dot(N, light_vector), 0.0), 2.0);
    vec3 ocean_color = base_color * (0.3 + 0.7 * diffuse);
    float F0 = 0.02;                   
    float fresnel = F0 + (1.0 - F0) * pow(1.0 - max(dot(N, V), 0.0), 5.0);
    fresnel = clamp(fresnel, 0.0, 1.0);
    vec3 color = mix(ocean_color, vec3(0.0, 0.0, 0.0), fresnel);  
    color += spec * vec3(1.0, 0.95, 0.85) * 0.6;          
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));    
    f_color = vec4(color, 1.0);
}