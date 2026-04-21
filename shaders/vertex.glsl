#version 330
uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;
uniform sampler2D height_map;
uniform sampler2D height_map_dx;
uniform sampler2D normal_map;
in vec2 in_pos; 
out float v_height;
out vec3 v_normal;
out vec3 v_pos;
void main() {
    vec2 ub = in_pos * 0.5 + 0.5;
    vec2 displacement = texture(height_map_dx, ub).gb;
    float y = texture(height_map, ub).r;
    vec2 grad = texture(normal_map, ub).rg;
    v_normal = normalize(vec3(-grad.x, 0.1, -grad.y));
    v_height = y;
    v_pos = vec3(in_pos.x + displacement.x, y, in_pos.y + displacement.y);
    gl_Position = proj * view * model * vec4(v_pos, 1.0);
}
