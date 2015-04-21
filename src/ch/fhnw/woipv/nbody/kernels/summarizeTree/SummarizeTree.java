package ch.fhnw.woipv.nbody.kernels.summarizeTree;

import java.io.File;
import java.io.IOException;

import ch.fhnw.woipv.nbody.internal.opencl.CLCommandQueue;
import ch.fhnw.woipv.nbody.internal.opencl.CLContext;
import ch.fhnw.woipv.nbody.internal.opencl.CLKernel;
import ch.fhnw.woipv.nbody.internal.opencl.CLMemory;
import ch.fhnw.woipv.nbody.internal.opencl.CLProgram;
import ch.fhnw.woipv.nbody.internal.opencl.CLProgram.BuildOption;
import ch.fhnw.woipv.nbody.kernels.NBodyKernel;

public class SummarizeTree implements NBodyKernel {
	
	private static final String BUILD_TREE_KERNEL_FILE = "kernels/nbody/summarizetree.cl";
	private static final String BUILD_TREE_KERNEL_NAME = "summarizeTree";
	
	public void summarizeTree(final CLContext context, final CLCommandQueue commandQueue, final CLMemory bodiesXBuffer, final CLMemory bodiesYBuffer,
			final CLMemory bodiesZBuffer, final CLMemory blockCountBuffer, final CLMemory radiusBuffer, final CLMemory bottomBuffer, final CLMemory massBuffer,
			final CLMemory childBuffer, int numberOfBodies, int globalWorkSize, int localWorkSize, int numWorkGroups, int numberOfNodes) throws IOException {

		final CLProgram program = context.createProgram(new File(BUILD_TREE_KERNEL_FILE));

		program.build(BuildOption.CL20, BuildOption.MAD,
				DEBUG,
				numberOfNodes(numberOfNodes),
				numberOfBodies(numberOfBodies),
				workgroupSize(localWorkSize),
				numberOfWorkgroups(numWorkGroups));

		final CLKernel kernel = program.createKernel(BUILD_TREE_KERNEL_NAME);

		kernel.addArgument(bodiesXBuffer);
		kernel.addArgument(bodiesYBuffer);
		kernel.addArgument(bodiesZBuffer);

		kernel.addArgument(blockCountBuffer);
		kernel.addArgument(radiusBuffer);
		kernel.addArgument(bottomBuffer);
		kernel.addArgument(massBuffer);
		kernel.addArgument(childBuffer);

		commandQueue.execute(kernel, 1, globalWorkSize, localWorkSize);

		commandQueue.finish();
	}
}
