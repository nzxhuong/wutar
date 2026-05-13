#version 330
uniform mat4 view;
uniform mat4 proj;
in vec3 in_pos;
out vec3 v_dir;
void main() {
    v_dir = (inverse(view) * vec4(in_pos, 0.0)).xyz;
    gl_Position = proj * view * vec4(in_pos, 1.0);
}