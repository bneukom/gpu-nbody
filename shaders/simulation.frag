#version 330

out vec4 outColor;
in vec4 velocity;

uniform sampler2D tex;

void main(void) {
	// outColor = texture(tex, gl_PointCoord) * vec4(251f/ 255f, 172f / 255f, 71f / 255f ,1) ;
	// outColor = vec4(1.0,1.0,1.0,1.0);
	outColor = texture(tex, gl_PointCoord) * vec4(clamp(length(velocity), 0, 1), clamp(1 - length(velocity) / 2, 0, 1), 0.25f, 1);
}