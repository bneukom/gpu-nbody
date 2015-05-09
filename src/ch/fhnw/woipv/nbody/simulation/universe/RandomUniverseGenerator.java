package ch.fhnw.woipv.nbody.simulation.universe;

public class RandomUniverseGenerator implements UniverseGenerator {

	private final float range;

	public RandomUniverseGenerator(float range) {
		this.range = range;
	}

	@Override
	public void generate(int offset, int nbodies, float[] bodiesX, float[] bodiesY, float[] bodiesZ, float[] velX, float[] velY, float[] velZ, float[] mass) {
		for (int i = 0; i < nbodies; ++i) {
			bodiesX[i + offset] = (float) ((Math.random() -0.5) * range);
			bodiesY[i + offset] = (float) ((Math.random() -0.5) * range);
			bodiesZ[i + offset] = (float) ((Math.random() -0.5) * range);
			mass[i + offset] = 1f / nbodies;
		}
	}
}
