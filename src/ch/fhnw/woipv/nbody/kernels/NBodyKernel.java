package ch.fhnw.woipv.nbody.kernels;

import ch.fhnw.woipv.nbody.internal.opencl.CLProgram.BuildOption;

public interface NBodyKernel {
	public static final BuildOption DEBUG = new BuildOption("-D DEBUG");

	public default BuildOption numberOfNodes(int numberOfNodes) {
		return new BuildOption("-D NUMBER_OF_NODES=" + numberOfNodes);
	}
	
	public default BuildOption numberOfBodies(int nbodies) {
		return new BuildOption("-D NBODIES=" + nbodies);
	}
	
	public default BuildOption workgroupSize(int localWorkSize) {
		return new BuildOption("-D WORKGROUP_SIZE=" + localWorkSize);
	}
	
	public default BuildOption numberOfWorkgroups(int numWorkGroups) {
		return new BuildOption("-D NUM_WORK_GROUPS=" + numWorkGroups);
	}
}
