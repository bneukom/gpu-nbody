package ch.fhnw.woipv.nbody.simulation;

import com.jogamp.opengl.GL3;

public interface NBodySimulation {
	public void step();

	void initPositionBuffer(GL3 gl, int vbo);

	public int getNumberOfBodies();
	
	public void init(GL3 gl);
}
