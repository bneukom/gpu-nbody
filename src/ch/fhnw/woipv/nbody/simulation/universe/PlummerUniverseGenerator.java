package ch.fhnw.woipv.nbody.simulation.universe;

public class PlummerUniverseGenerator implements UniverseGenerator {

	@Override
	public void generate(int offset, int nbodies, float[] bodiesX, float[] bodiesY, float[] bodiesZ, float[] velX, float[] velY, float[] velZ, float[] bodiesMass) {

		double rsc = (3 * Math.PI) / 16;
		double vsc = Math.sqrt(1.0 / rsc);
		for (int i = 0; i < nbodies; i++) {
			bodiesMass[i] = (float) (1.0 / nbodies);
			double r = 1.0 / Math.sqrt(Math.pow(Math.random() * 0.999, -2.0 / 3.0) - 1);

			double x, y, z, sq;
			do {
				x = Math.random() * 2.0 - 1.0;
				y = Math.random() * 2.0 - 1.0;
				z = Math.random() * 2.0 - 1.0;
				sq = x * x + y * y + z * z;
			} while (sq > 1.0);
			double scale = rsc * r / Math.sqrt(sq);
			bodiesX[i] = (float) (x * scale);
			bodiesY[i] = (float) (y * scale);
			bodiesZ[i] = (float) (z * scale);

			do {
				x = Math.random();
				y = Math.random() * 0.1;
			} while (y > x * x * Math.pow(1 - x * x, 3.5));
			double v = x * Math.sqrt(2.0 / Math.sqrt(1 + r * r));
			do {
				x = Math.random() * 2.0 - 1.0;
				y = Math.random() * 2.0 - 1.0;
				z = Math.random() * 2.0 - 1.0;
				sq = x * x + y * y + z * z;
			} while (sq > 1.0);
			scale = vsc * v / Math.sqrt(sq);
			velX[i] = (float) (x * scale);
			velY[i] = (float) (y * scale);
			velZ[i] = (float) (z * scale);
		}

	}

}
