package ch.fhnw.woipv.nbody.simulation.generators;

public class RandomUniverseGenerator implements UniverseGenerator {

	private final float range;

	public RandomUniverseGenerator(float range) {
		this.range = range;
	}

	@Override
	public void generate(int offset, int nbodies, float[] bodiesX, float[] bodiesY, float[] bodiesZ, float[] mass) {
		for (int i = 0; i < nbodies; ++i) {
			bodiesX[i + offset] = (float) (Math.random() * range);
			bodiesY[i + offset] = (float) (Math.random() * range);
			bodiesZ[i + offset] = (float) (Math.random() * range);
			mass[i + offset] = 1;
		}
	}

}
