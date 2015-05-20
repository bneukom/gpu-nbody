package ch.fhnw.woipv.nbody.simulation.universe.serialize;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;

import ch.fhnw.woipv.nbody.simulation.universe.UniverseGenerator;

public class SerializedUniverseGenerator implements UniverseGenerator {

	private float[] bodiesX;
	private float[] bodiesY;
	private float[] bodiesZ;
	private float[] velX;
	private float[] velY;
	private float[] velZ;
	private float[] bodiesMass;

	private int nbodies;

	public SerializedUniverseGenerator(final String file) {
		try (ObjectInputStream inputStream = new ObjectInputStream(new FileInputStream(file))) {
			nbodies = inputStream.readInt();

			bodiesX = (float[]) inputStream.readObject();
			bodiesY = (float[]) inputStream.readObject();
			bodiesZ = (float[]) inputStream.readObject();
			velX = (float[]) inputStream.readObject();
			velY = (float[]) inputStream.readObject();
			velZ = (float[]) inputStream.readObject();
			bodiesMass = (float[]) inputStream.readObject();

		} catch (IOException | ClassNotFoundException e) {
			e.printStackTrace();
		}

	}

	@Override
	public void generate(int offset, int nbodies, float[] bodiesX, float[] bodiesY, float[] bodiesZ, float[] velX, float[] velY, float[] velZ, float[] bodiesMass) {
		if (nbodies != this.nbodies)
			throw new IllegalStateException("invalid amount of bodies for serialized universe");

		for (int body = 0; body < nbodies; ++body) {
			bodiesX[body + offset] = this.bodiesX[body];
			bodiesY[body + offset] = this.bodiesY[body];
			bodiesZ[body + offset] = this.bodiesZ[body];
			velX[body + offset] = this.velX[body];
			velY[body + offset] = this.velY[body];
			velZ[body + offset] = this.velZ[body];
			bodiesMass[body + offset] = this.bodiesMass[body];
		}
	}

}
