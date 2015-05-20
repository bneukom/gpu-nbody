package ch.fhnw.woipv.nbody.simulation.universe;

/**
 * Generates a certain universe
 *
 */
public interface UniverseGenerator {
	/**
	 * Generates the universe and writes its values to the given fields.
	 * 
	 * @param offset
	 *            the offset inside the given arrays
	 * @param nbodies
	 * @param bodiesX
	 * @param bodiesY
	 * @param bodiesZ
	 * @param velX
	 * @param velY
	 * @param velZ
	 * @param bodiesMass
	 */
	public void generate(int offset, int nbodies, float[] bodiesX, float bodiesY[], float bodiesZ[], float[] velX, float[] velY, float[] velZ, float[] bodiesMass);
}
