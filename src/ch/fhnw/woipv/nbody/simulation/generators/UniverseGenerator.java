package ch.fhnw.woipv.nbody.simulation.generators;

public interface UniverseGenerator {
	public void generate(int offset, int nbodies, float[] bodiesX, float bodiesY[], float bodiesZ[], float[] bodiesMass);
}
