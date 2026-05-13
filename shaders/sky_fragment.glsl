#version 330
uniform samplerCube skybox;
in vec3 v_dir;
out vec4 f_color;
void main() {
    f_color = texture(skybox, v_dir);
}