#version 330
in vec2 v_uv;
out vec4 f_color;
uniform sampler2D screen_tex;
void main() {
    f_color = texture(screen_tex, v_uv);
}
