package ch.fhnw.woipv.nbody.kernels.boundsReduction;

import ch.fhnw.woipv.nbody.kernels.NBodyKernel;

public class BoundingBoxReduction implements NBodyKernel {

	@Override
	public String getKernelName() {
		return "boundingBox";
	}

	@Override
	public String getFileName() {
		return "kernels/nbody/boundingbox.cl";
	}
}
