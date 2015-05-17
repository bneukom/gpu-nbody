package ch.fhnw.woipv.nbody.simulation.universe.test;

import ch.fhnw.woipv.nbody.simulation.universe.UniverseGenerator;

public class TwoBodyUniverse implements UniverseGenerator{

	@Override
	public void generate(int offset, int nbodies, float[] bodiesX, float[] bodiesY, float[] bodiesZ, float[] velX, float[] velY, float[] velZ, float[] bodiesMass) {
		bodiesX[0] = -1.1f;
		bodiesY[0] = -1;
		bodiesZ[0] = -1;
		bodiesMass[0] = 1 / 8f;

		bodiesX[1] = 1;
		bodiesY[1] = 1.1f;
		bodiesZ[1] = 1;
		bodiesMass[1] = 1 / 8f;
	}

}
