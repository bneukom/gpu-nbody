package ch.fhnw.woipv.nbody.simulation.universe.test;

import ch.fhnw.woipv.nbody.simulation.universe.UniverseGenerator;

public class BigTreeUniverse implements UniverseGenerator {

	@Override
	public void generate(int offset, int nbodies, float[] bodiesX, float[] bodiesY, float[] bodiesZ, float[] velX, float[] velY, float[] velZ, float[] bodiesMass) {
		bodiesX[0] = 0;
		bodiesY[0] = 0;
		bodiesZ[0] = 0;
		bodiesMass[0] = 0.1f;
		bodiesX[1] = 0.00000001f;
		bodiesY[1] = 0;
		bodiesZ[1] = 0;
		bodiesMass[1] = 0.1f;
		bodiesX[2] = 10000f;
		bodiesY[2] = 10000f;
		bodiesZ[2] = 10000f;
		bodiesX[3] = -10000f;
		bodiesY[3] = -10000f;
		bodiesZ[3] = -10000f;
		bodiesMass[2] = 0.1f;
	}

}
