package ch.fhnw.woipv.nbody.simulation.universe.serialize;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;

import ch.fhnw.woipv.nbody.simulation.universe.MonteCarloSphericalUniverseGenerator;
import ch.fhnw.woipv.nbody.simulation.universe.UniverseGenerator;

public class UniverseSerializer {
	public static void main(final String[] args) {
		final UniverseGenerator generator = new MonteCarloSphericalUniverseGenerator();
		final int offset = 0;
		final int nbodies = 2048 * 16;
		final float[] bodiesX = new float[nbodies];
		final float[] bodiesY = new float[nbodies];
		final float[] bodiesZ = new float[nbodies];
		final float[] velX = new float[nbodies];
		final float[] velY = new float[nbodies];
		final float[] velZ = new float[nbodies];
		final float[] bodiesMass = new float[nbodies];
		
		generator.generate(offset, nbodies, bodiesX, bodiesY, bodiesZ, velX, velY, velZ, bodiesMass);
		
		try (ObjectOutputStream serializer = new ObjectOutputStream(new FileOutputStream("universes/montecarlouniverse1.universe"))) {
			
			serializer.writeInt(nbodies);
			serializer.writeObject(bodiesX);
			serializer.writeObject(bodiesY);
			serializer.writeObject(bodiesZ);
			serializer.writeObject(velX);
			serializer.writeObject(velY);
			serializer.writeObject(velZ);
			serializer.writeObject(bodiesMass);
			
		} catch (final IOException e) {
			e.printStackTrace();
		}
	}
}
