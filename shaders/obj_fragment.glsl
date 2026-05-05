#version 330
in vec3 v_norm;
out vec4 f_color;
uniform vec3 light_vector;
void main() {
    float diff = max(dot(normalize(v_norm), light_vector), 0.3);
    f_color = vec4(vec3(0.6, 0.3, 0.1) * diff, 1.0);
}
