#version 330
uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;
uniform float WORLD_L;
uniform sampler2D height_map;
uniform sampler2D height_map_dx;
in vec2 in_pos; 
out float v_height;
out vec2 v_uv;
out vec3 v_pos;
void main() {
    vec2 uv = (in_pos / WORLD_L) + 0.5;
    v_uv = uv;
    vec2 displacement = texture(height_map_dx, uv).xy;
    float y = texture(height_map, uv).r;
    v_pos = vec3(in_pos.x + displacement.x, y, in_pos.y + displacement.y);
    v_height = y;
    gl_Position = proj * view * model * vec4(v_pos, 1.0);
}
