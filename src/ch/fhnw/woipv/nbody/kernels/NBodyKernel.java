package ch.fhnw.woipv.nbody.kernels;

import java.io.File;
import java.io.IOException;

import net.benjaminneukom.oocl.cl.CLCommandQueue;
import net.benjaminneukom.oocl.cl.CLContext;
import net.benjaminneukom.oocl.cl.CLKernel;
import net.benjaminneukom.oocl.cl.CLMemory;
import net.benjaminneukom.oocl.cl.CLProgram;
import net.benjaminneukom.oocl.cl.CLProgram.BuildOption;

public interface NBodyKernel {

	public default void execute(final CLContext context, final CLCommandQueue commandQueue, 
			final CLMemory bodiesXBuffer, final CLMemory bodiesYBuffer, final CLMemory bodiesZBuffer, 
			final CLMemory blockCountBuffer, final CLMemory radiusBuffer, final CLMemory bottomBuffer, final CLMemory massBuffer,
			final CLMemory childBuffer, CLMemory bodyCountBuffer, int numberOfBodies, int globalWorkSize, int localWorkSize, int numWorkGroups, int numberOfNodes, int warpSize,
			boolean debug) throws IOException {

		final CLProgram program = context.createProgram(new File(getFileName()));

		program.build(BuildOption.CL20, BuildOption.MAD,
				debug ? DEBUG : null,
				numberOfNodes(numberOfNodes),
				numberOfBodies(numberOfBodies),
				workgroupSize(localWorkSize),
				numberOfWorkgroups(numWorkGroups));

		final CLKernel kernel = program.createKernel(getKernelName());

		kernel.addArgument(bodiesXBuffer);
		kernel.addArgument(bodiesYBuffer);
		kernel.addArgument(bodiesZBuffer);

		kernel.addArgument(blockCountBuffer);
		kernel.addArgument(bodyCountBuffer);
		kernel.addArgument(radiusBuffer);
		kernel.addArgument(bottomBuffer);
		kernel.addArgument(massBuffer);
		kernel.addArgument(childBuffer);

		commandQueue.execute(kernel, 1, globalWorkSize, localWorkSize);

		commandQueue.finish();
	}

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

	/**
	 * Returns the name of the kernel to be executed.
	 * 
	 * @return
	 */
	public String getKernelName();

	/**
	 * Returns the file where the kernel resides.
	 * 
	 * @return
	 */
	public String getFileName();
}
