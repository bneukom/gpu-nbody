#version 330
in vec4 inVertex;
in vec3 inColor;
uniform mat4 modelviewMatrix;
uniform mat4 projectionMatrix;
void main(void) {
	gl_Position =  projectionMatrix * modelviewMatrix * inVertex;
	
}
