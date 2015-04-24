package ch.fhnw.woipv.nbody.kernels.summarizeTree;

import ch.fhnw.woipv.nbody.kernels.NBodyKernel;

public class SummarizeTree implements NBodyKernel {

	private static final String BUILD_TREE_KERNEL_FILE = "kernels/nbody/summarizetree.cl";
	private static final String BUILD_TREE_KERNEL_NAME = "summarizeTree";

	@Override
	public String getKernelName() {
		return BUILD_TREE_KERNEL_NAME;
	}

	@Override
	public String getFileName() {
		return BUILD_TREE_KERNEL_FILE;
	}

}
