package ch.fhnw.woipv.nbody.simulation.universe.test;

import ch.fhnw.woipv.nbody.simulation.universe.UniverseGenerator;

public class EightBodyUniverse implements UniverseGenerator{

	@Override
	public void generate(int offset, int nbodies, float[] bodiesX, float[] bodiesY, float[] bodiesZ, float[] velX, float[] velY, float[] velZ, float[] bodiesMass) {
		bodiesX[0] = -1.1f;
		bodiesY[0] = -1;
		bodiesZ[0] = -1;
		bodiesMass[0] = 1 / 8f;

		bodiesX[1] = 1;
		bodiesY[1] = -1;
		bodiesZ[1] = -1;
		bodiesMass[1] = 1 / 8f;
		
		bodiesX[2] = -1;
		bodiesY[2] = 1;
		bodiesZ[2] = -1.1f;
		bodiesMass[2] = 1 / 8f;

		bodiesX[3] = -1;
		bodiesY[3] = -1;
		bodiesZ[3] = 1.1f;
		bodiesMass[3] = 1 / 8f;
		
		bodiesX[4] = 1;
		bodiesY[4] = 1;
		bodiesZ[4] = -1;
		bodiesMass[4] = 1 / 8f;

		bodiesX[5] = -1;
		bodiesY[5] = 1.1f;
		bodiesZ[5] = 1;
		bodiesMass[5] = 1 / 8f;
		
		bodiesX[6] = 1;
		bodiesY[6] = -1;
		bodiesZ[6] = 1;
		bodiesMass[6] = 1 / 8f;

		bodiesX[7] = 1;
		bodiesY[7] = 1.1f;
		bodiesZ[7] = 1;
		bodiesMass[7] = 1 / 8f;
	}

}
