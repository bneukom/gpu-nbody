#version 330

in vec4 inVertex;
in vec4 inVelocity;
out vec4 velocity;

uniform mat4 modelviewMatrix;
uniform mat4 projectionMatrix;
uniform vec2 screenSize;
uniform float spriteSize;

void main(void) {
	// gl_Position =  projectionMatrix * modelviewMatrix * inVertex;
	vec4 eyePos = modelviewMatrix * inVertex;
    vec4 projVoxel = projectionMatrix * vec4(spriteSize,spriteSize,eyePos.z,eyePos.w);
    vec2 projSize = screenSize * projVoxel.xy / projVoxel.w;
    
    gl_PointSize = 0.25 * (projSize.x+projSize.y);
    gl_Position = projectionMatrix * eyePos;
    
    velocity = inVelocity;
}
