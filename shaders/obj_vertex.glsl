#version 330
uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;
in vec3 in_vert;
in vec3 in_norm;
out vec3 v_norm;
void main() {
    v_norm = mat3(model) * in_norm;
    gl_Position = proj * view * model * vec4(in_vert, 1.0);
}
