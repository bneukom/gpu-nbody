package ch.fhnw.woipv.nbody.kernels.calculateForce;

import ch.fhnw.woipv.nbody.kernels.NBodyKernel;

public class CalculateForce implements NBodyKernel {

	private static final String CALCULATE_FORCE_KERNEL_FILE = "kernels/nbody/calculateforce.cl";
	private static final String CALCULATE_FORCE_KERNEL_NAME = "calculateForce";

	@Override
	public String getKernelName() {
		return CALCULATE_FORCE_KERNEL_NAME;
	}

	@Override
	public String getFileName() {
		return CALCULATE_FORCE_KERNEL_FILE;
	}

}
