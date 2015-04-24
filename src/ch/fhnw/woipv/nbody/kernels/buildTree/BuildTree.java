package ch.fhnw.woipv.nbody.kernels.buildTree;

import ch.fhnw.woipv.nbody.kernels.NBodyKernel;

public class BuildTree implements NBodyKernel {

	@Override
	public String getKernelName() {
		return "buildTree";
	}

	@Override
	public String getFileName() {
		// TODO Auto-generated method stub
		return "kernels/nbody/buildtree.cl";
	}
}
