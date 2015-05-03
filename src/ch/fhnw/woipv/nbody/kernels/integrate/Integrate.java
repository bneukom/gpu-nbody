package ch.fhnw.woipv.nbody.kernels.integrate;

import ch.fhnw.woipv.nbody.kernels.NBodyKernel;

public class Integrate implements NBodyKernel {

	private static final String INTEGRATE_KERNEL_FILE = "kernels/nbody/integrate.cl";
	private static final String INTEGRATE_KERNEL_NAME = "integrate";

	@Override
	public String getKernelName() {
		return INTEGRATE_KERNEL_NAME;
	}

	@Override
	public String getFileName() {
		return INTEGRATE_KERNEL_FILE;
	}

}
