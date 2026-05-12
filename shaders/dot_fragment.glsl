#version 330
uniform vec3 u_color;
out vec4 f_color;
void main() {
    vec2 c = gl_PointCoord - 0.5;
    float r = dot(c, c);
    // Solid filled circle with soft anti-aliased edge
    float alpha = 1.0 - smoothstep(0.20, 0.25, r);
    if (alpha < 0.01) discard;
    f_color = vec4(u_color, alpha);
}
