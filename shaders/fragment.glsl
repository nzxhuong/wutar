#version 330
in float v_height;
in vec3 v_normal;
in vec3 v_pos;
out vec4 f_color;
uniform vec3 light_vector;
uniform vec3 cam_pos;
void main() {
    vec3 V = normalize(cam_pos - v_pos);
    vec3 H = normalize(light_vector + V);
    float spec = pow(max(dot(v_normal, H), 0.0), 32.0);
    float h = clamp(v_height * 10.0, 0.0, 1.0);
    vec3 color = vec3(0.1, 0.3 + h*0.5, 0.6 + h*0.4); 
    float diffuse = max(dot(v_normal, light_vector), 0.0);
    color *= 0.3 + 0.7 * diffuse;
    color += spec * 0.5;
    f_color = vec4(color, 1.0);
}
