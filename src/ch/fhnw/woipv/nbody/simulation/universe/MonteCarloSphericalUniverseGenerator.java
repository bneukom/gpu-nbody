package ch.fhnw.woipv.nbody.simulation.universe;

public class MonteCarloSphericalUniverseGenerator implements UniverseGenerator {

	@Override
	public void generate(int offset, int nbodies, float[] bodiesX, float[] bodiesY, float[] bodiesZ, float[] velX, float[] velY, float[] velZ, float[] bodiesMass) {
		final double rand = Math.random() * nbodies;

		double o = 2 / (double) nbodies;
		double increment = Math.PI * (3.0 - Math.sqrt(5));

		for (int i = 0; i < nbodies; ++i) {
			double y = ((i * o) - 1) + (o / 2);
			double r = Math.sqrt(1 - Math.pow(y, 2));

			double phi = ((i + rand) % nbodies) * increment;

			double x = Math.cos(phi) * r;
			double z = Math.sin(phi) * r;

			bodiesX[i] = (float) x;
			bodiesY[i] = (float) y;
			bodiesZ[i] = (float) z;
			bodiesMass[i] = 1f / nbodies;
		}

	}
	// for (int i = 1; i <= min(nbrPoints,pts.length); ++i) {
	// float lon = ga*i;
	// lon /= 2*PI; lon -= floor(lon); lon *= 2*PI;
	// if (lon > PI) lon -= 2*PI;
	//
	// // Convert dome height (which is proportional to surface area) to latitude
	// float lat = asin(-1 + 2*i/(float)nbrPoints);
	//
	// pts[i] = new SpherePoint(lat, lon);
	// }
}
