package ch.fhnw.woipv.nbody.simulation.universe;

public class RotatingDiskGalaxyGenerator implements UniverseGenerator {
	
	private float radius;
	private float velocityMultiplier;
	private float centerMass;

	public RotatingDiskGalaxyGenerator(final float r, final float velocityMultiplier, final float centerMass) {
		this.radius = r;
		this.velocityMultiplier = velocityMultiplier;
		this.centerMass = centerMass;
	}

	@Override
	public void generate(int offset, int nbodies, float[] bodiesX, float[] bodiesY, float[] bodiesZ, float[] velX, float[] velY, float[] velZ, float[] bodiesMass) {

		bodiesMass[0] = centerMass;

		for (int i = 1; i < nbodies; ++i) {
			float r = (float) (Math.random() * radius) + 0.1f; // + 0.1 to ensure it won't be on the center
			double alpha = Math.random() * 2 * Math.PI;
			float x = (float) (Math.cos(alpha) * r);
			float y = (float) (Math.sin(alpha) * r);

			bodiesX[i] = x;
			bodiesY[i] = y;
			bodiesZ[i] = (float) ((Math.random() - 0.5) / 8);

			bodiesMass[i] = 1f / nbodies;

			// orbital velocity
			final float v0 = (float) Math.sqrt((bodiesMass[0] + bodiesMass[i]) / (r * r * r)) * velocityMultiplier;

			// rotate by 90°
			final float vx = y * v0;
			final float vy = -x * v0;

			velX[i] = vx;
			velY[i] = vy;
		}
	}

}
