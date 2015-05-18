package ch.fhnw.woipv.nbody.simulation.cpu;

import static com.jogamp.opengl.GL.*;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import ch.fhnw.woipv.nbody.simulation.AbstractNBodySimulation;
import ch.fhnw.woipv.nbody.simulation.universe.UniverseGenerator;

import com.jogamp.opengl.GL3;

public class CPUBruteForceNBodySimulation extends AbstractNBodySimulation {

	private int vbo;
	private GL3 gl;

	private float[] bodyX;
	private float[] bodyY;
	private float[] bodyZ;

	private float[] accX;
	private float[] accY;
	private float[] accZ;

	private float[] velX;
	private float[] velY;
	private float[] velZ;

	private float[] mass;

	private final int nbodies;
	private UniverseGenerator generator;

	private static final float EPSILON = 0.05f * 0.05f;
	private static final float TIMESTEP = 0.025f;

	public CPUBruteForceNBodySimulation(Mode mode, int nbodies, UniverseGenerator generator) {
		super(mode);
		this.generator = generator;
		this.nbodies = nbodies;

		this.bodyX = new float[nbodies];
		this.bodyY = new float[nbodies];
		this.bodyZ = new float[nbodies];

		this.accX = new float[nbodies];
		this.accY = new float[nbodies];
		this.accZ = new float[nbodies];

		this.velX = new float[nbodies];
		this.velY = new float[nbodies];
		this.velZ = new float[nbodies];

		this.mass = new float[nbodies];
	}

	@Override
	public void step() {
		// simulate
		for (int i = 0; i < nbodies; ++i) {
			float m1 = mass[i];
			for (int j = i + 1; j < nbodies; ++j) {
				if (i == j)
					continue;

				float m2 = mass[j];

				float dx = bodyX[j] - bodyX[i];
				float dy = bodyY[j] - bodyY[i];
				float dz = bodyZ[j] - bodyZ[i];

				float dist = (float) Math.sqrt(dx * dx + dy * dy + dz * dz) + EPSILON;

				double f = (m1 * m2) / (dist * dist * dist);

				accX[i] += dx * f / m1;
				accY[i] += dy * f / m1;
				accZ[i] += dz * f / m1;

				accX[j] += -dx * f / m2;
				accY[j] += -dy * f / m2;
				accZ[j] += -dz * f / m2;

			}
		}

		// integrate
		for (int i = 0; i < nbodies; ++i) {
			velX[i] += TIMESTEP * accX[i];
			velY[i] += TIMESTEP * accY[i];
			velZ[i] += TIMESTEP * accZ[i];

			bodyX[i] += TIMESTEP * velX[i];
			bodyY[i] += TIMESTEP * velY[i];
			bodyZ[i] += TIMESTEP * velZ[i];
		}

		// copy to opengl
		if (mode == Mode.GL_INTEROP) {
			gl.glBindBuffer(GL_ARRAY_BUFFER, vbo);

			final ByteBuffer byteBuffer = gl.glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
			final FloatBuffer vertices = byteBuffer.order(ByteOrder.nativeOrder()).asFloatBuffer();

			for (int i = 0; i < nbodies; ++i) {
				final int index = 4 * i;
				vertices.put(index + 0, bodyX[i]);
				vertices.put(index + 1, bodyY[i]);
				vertices.put(index + 2, bodyZ[i]);
				vertices.put(index + 3, 1);
			}

			gl.glUnmapBuffer(GL_ARRAY_BUFFER);
		}
		
		Thread.yield();

		// printEnergy();
	}

	private void printEnergy() {
		double kineticEnergy = 0;
		double potentialEnergy = 0;

		for (int i = 0; i < nbodies; ++i) {
			final float vx = velX[i];
			final double vy = velY[i];
			final double vz = velZ[i];
			final double v = Math.sqrt(vx * vx + vy * vy + vz * vz);

			final double m1 = mass[i];
			kineticEnergy += (m1 * v * v / 2);

			for (int j = 0; j < nbodies; ++j) {
				if (i == j)
					continue;

				final double m2 = mass[j];
				final double deltaX = bodyX[i] - bodyX[j];
				final double deltaY = bodyY[i] - bodyY[j];
				final double deltaZ = bodyZ[i] - bodyZ[j];
				final double r = Math.sqrt(deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ);
				double e = (m1 * m2) / (r * r * r);

				potentialEnergy += e;

			}
		}

		System.out.println("ekin: " + kineticEnergy + " epot:" + potentialEnergy + " etot: " + (kineticEnergy - potentialEnergy));
	}

	@Override
	public void initPositionBuffer(GL3 gl, int vbo) {
		this.vbo = vbo;

	}

	@Override
	public int getNumberOfBodies() {
		return nbodies;
	}

	@Override
	public void init(GL3 gl) {
		this.gl = gl;
		this.generator.generate(0, nbodies, bodyX, bodyY, bodyZ, velX, velY, velZ, mass);
	}

}
