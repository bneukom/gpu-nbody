package ch.fhnw.woipv.nbody.kernels.buildTree;

import java.io.File;
import java.io.IOException;

import ch.fhnw.woipv.nbody.internal.opencl.CLCommandQueue;
import ch.fhnw.woipv.nbody.internal.opencl.CLContext;
import ch.fhnw.woipv.nbody.internal.opencl.CLKernel;
import ch.fhnw.woipv.nbody.internal.opencl.CLMemory;
import ch.fhnw.woipv.nbody.internal.opencl.CLProgram;
import ch.fhnw.woipv.nbody.internal.opencl.CLProgram.BuildOption;

public class BuildTree {
	public void buildTree(final CLContext context, final CLCommandQueue commandQueue, final CLMemory bodiesXBuffer, final CLMemory bodiesYBuffer,
			final CLMemory bodiesZBuffer, final CLMemory blockCountBuffer, final CLMemory radiusBuffer, final CLMemory bottomBuffer, final CLMemory massBuffer,
			final CLMemory childBuffer, int numberOfBodies, int globalWorkSize, int localWorkSize, int numWorkGroups, int numberOfNodes) throws IOException {

		final CLProgram program = context.createProgram(new File("kernels/nbody/buildtree.cl"));

		program.build(BuildOption.CL20, BuildOption.MAD,
//				new BuildOption("-g"),
//				new BuildOption("-s C:/dev/workspace-gpunbody/JoclNBody/kernels/nbody/boundingbox.cl"),
//				new BuildOption("-D DEBUG"),
				new BuildOption("-D NUMBER_OF_NODES=" + numberOfNodes),
				new BuildOption("-D NBODIES=" + numberOfBodies),
				new BuildOption("-D WORKGROUP_SIZE=" + localWorkSize),
				new BuildOption("-D NUM_WORK_GROUPS=" + numWorkGroups));

		final CLKernel kernel = program.createKernel("buildTree");

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
