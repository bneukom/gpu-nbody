package ch.fhnw.woipv.nbody.simulation.universe;

public interface UniverseGenerator {
	public void generate(int offset, int nbodies, float[] bodiesX, float bodiesY[], float bodiesZ[], float[] velX, float[] velY, float[] velZ, float[] bodiesMass);
}
