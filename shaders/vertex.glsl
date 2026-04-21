#version 330
uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;
uniform sampler2D height_map;
in vec2 in_pos; 
out float v_height;
out vec3 v_normal;
out vec3 v_pos;
void main() {
    vec2 ub = in_pos * 0.5 + 0.5;
    float y = texture(height_map, ub).r;
    float sm = 1.0 / 1024.0;
    float lx = texture(height_map, ub + vec2(-sm, 0.0)).r;
    float ry = texture(height_map, ub + vec2(sm, 0.0)).r;
    float lz = texture(height_map, ub + vec2(0.0, -sm)).r;
    float tz = texture(height_map, ub + vec2(0.0, sm)).r;

    v_normal = normalize(vec3(lx - ry, 0.1, lz - tz));
    v_height = y;
    v_pos = vec3(in_pos.x, y, in_pos.y);
    gl_Position = proj * view * model * vec4(in_pos.x, y, in_pos.y, 1.0);
    
}
