package ch.fhnw.woipv.nbody.kernels.sort;

import ch.fhnw.woipv.nbody.kernels.NBodyKernel;

public class Sort implements NBodyKernel {

	private static final String SORT_KERNEL_FILE = "kernels/nbody/sort.cl";
	private static final String SORT_KERNEL_NAME = "sort";

	@Override
	public String getKernelName() {
		return SORT_KERNEL_NAME;
	}

	@Override
	public String getFileName() {
		return SORT_KERNEL_FILE;
	}

}
