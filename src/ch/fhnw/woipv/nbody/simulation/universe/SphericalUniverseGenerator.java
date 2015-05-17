package ch.fhnw.woipv.nbody.simulation.universe;

public class SphericalUniverseGenerator implements UniverseGenerator {
	private static final double R = 6.0;

	@Override
	public void generate(int offset, int nbodies, float[] bodiesX, float[] bodiesY, float[] bodiesZ, float[] velX, float[] velY, float[] velZ, float[] bodiesMass) {
		for (int i = 0; i < nbodies; ++i) {
			final double omega = Math.random() * 2 * Math.PI;
			final double u = Math.random() * 2 - 1;
			final double x = Math.sqrt(1 - u * u) * Math.cos(omega);
			final double y = Math.sqrt(1 - u * u) * Math.sin(omega);
			final double z = u;
			
			bodiesX[i] = (float) x;
			bodiesY[i] = (float) y;
			bodiesZ[i] = (float) z;
			bodiesMass[i] = 1f / nbodies;

		}
	}
}
