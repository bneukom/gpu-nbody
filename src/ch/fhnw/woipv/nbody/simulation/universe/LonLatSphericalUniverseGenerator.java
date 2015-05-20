package ch.fhnw.woipv.nbody.simulation.universe;

// http://stackoverflow.com/questions/1185408/converting-from-longitude-latitude-to-cartesian-coordinates
public class LonLatSphericalUniverseGenerator implements UniverseGenerator {

	private static final double R = 5.0;

	@Override
	public void generate(int offset, int nbodies, float[] bodiesX, float[] bodiesY, float[] bodiesZ, float[] velX, float[] velY, float[] velZ, float[] bodiesMass) {
		for (int i = 0; i < nbodies; ++i) {
			final double lon = Math.random() * 2 * Math.PI;
			final double lat = Math.random() * 2 * Math.PI;

			final double x = R * Math.cos(lat) * Math.cos(lon);
			final double y = R * Math.cos(lat) * Math.sin(lon);
			final double z = R * Math.sin(lat);

			bodiesX[i] = (float) x;
			bodiesY[i] = (float) y;
			bodiesZ[i] = (float) z;

			// if (Math.random() < 0.001f)
			// bodiesMass[i] = (float) (Math.random() * 0.1f + 0.1f);
			// else
			bodiesMass[i] = 1f / nbodies;

		}

	}
}
