package ch.fhnw.woipv.nbody.simulation;

import com.jogamp.opengl.GL3;

public abstract class AbstractNBodySimulation {

	public final Mode mode;

	public AbstractNBodySimulation(Mode mode) {
		this.mode = mode;
	}

	/**
	 * Executes the next step of the simulation.
	 */
	public abstract void step();

	/**
	 * Initializes the OpenGL position buffer for the bodies.
	 * 
	 * @param gl
	 * @param positionVBO
	 * @param velocityVBO 
	 */
	public abstract void initGLBuffers(GL3 gl, int positionVBO, int velocityVBO);

	/**
	 * Returns the number of bodies for this simulation.
	 * 
	 * @return
	 */
	public abstract int getNumberOfBodies();

	/**
	 * Initializes the simulation.
	 * 
	 * @param gl
	 */
	public abstract void init(GL3 gl);
	
	public enum Mode {
		GL_INTEROP, DEFAULT
	}
}
