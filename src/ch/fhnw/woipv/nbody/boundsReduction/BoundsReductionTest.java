package ch.fhnw.woipv.nbody.boundsReduction;

import static org.jocl.CL.*;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import ch.fhnw.woipv.nbody.internal.opencl.CL20;
import ch.fhnw.woipv.nbody.internal.opencl.CLCommandQueue;
import ch.fhnw.woipv.nbody.internal.opencl.CLContext;
import ch.fhnw.woipv.nbody.internal.opencl.CLDevice;
import ch.fhnw.woipv.nbody.internal.opencl.CLKernel;
import ch.fhnw.woipv.nbody.internal.opencl.CLMemory;
import ch.fhnw.woipv.nbody.internal.opencl.CLProgram;
import ch.fhnw.woipv.nbody.internal.opencl.CLProgram.BuildOption;

public class BoundsReductionTest {

	private static final int NUMBER_OF_BODIES = 10000;
	private static final float BODIES_RANGE = 1000;

	private static final int WORK_GROUPS = 80;
	private static final int LOCAL_WORK_SIZE = 100;
	private static final int GLOBAL_WORK_SIZE = WORK_GROUPS * LOCAL_WORK_SIZE;

	public static void main(String[] args) throws IOException {
		final CLDevice device = CL20.createDevice();

		final CLContext context = device.createContext();
		final CLCommandQueue commandQueue = context.createCommandQueue();

		final CLProgram program = context.createProgram(new File("kernels/nbody/boundingbox.cl"));

		program.build(BuildOption.CL20, BuildOption.MAD, new BuildOption("-D NBODIES=" + NUMBER_OF_BODIES), new BuildOption("-D BOUNDING_BOX_WORKGROUP_SIZE=" + LOCAL_WORK_SIZE));

		final CLKernel kernel = program.createKernel("boundingBox");

		final float bodiesX[] = new float[NUMBER_OF_BODIES];
		final float bodiesY[] = new float[NUMBER_OF_BODIES];
		final float bodiesZ[] = new float[NUMBER_OF_BODIES];

		final float minX[] = new float[WORK_GROUPS];
		final float minY[] = new float[WORK_GROUPS];
		final float minZ[] = new float[WORK_GROUPS];

		final float maxX[] = new float[WORK_GROUPS];
		final float maxY[] = new float[WORK_GROUPS];
		final float maxZ[] = new float[WORK_GROUPS];

		final int blockCount[] = new int[1];

		generateBodies(bodiesX, bodiesY, bodiesZ);

		System.out.println("HOST");
		// for (int i = 0; i < NUMBER_OF_BODIES; ++i) {
		// System.out.println("(" + bodiesX[i] + "," + bodiesY[i] + "," +
		// bodiesZ[i] + ")");
		// }
		System.out.println();

		final CLMemory bodiesXBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bodiesX);
		final CLMemory bodiesYBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bodiesY);
		final CLMemory bodiesZBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bodiesZ);

		final CLMemory minXBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, minX);
		final CLMemory minYBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, minY);
		final CLMemory minZBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, minZ);

		final CLMemory maxXBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, maxX);
		final CLMemory maxYBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, maxY);
		final CLMemory maxZBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, maxZ);

		final CLMemory blockCountBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, blockCount);

		kernel.addArgument(bodiesXBuffer);
		kernel.addArgument(bodiesYBuffer);
		kernel.addArgument(bodiesZBuffer);

		kernel.addArgument(minXBuffer);
		kernel.addArgument(minYBuffer);
		kernel.addArgument(minZBuffer);

		kernel.addArgument(maxXBuffer);
		kernel.addArgument(maxYBuffer);
		kernel.addArgument(maxZBuffer);

		kernel.addArgument(blockCountBuffer);

		System.out.println("DEVICE");
		commandQueue.execute(kernel, 1, GLOBAL_WORK_SIZE, LOCAL_WORK_SIZE);

		commandQueue.finish();

		commandQueue.readBuffer(minXBuffer);
		commandQueue.readBuffer(minYBuffer);
		commandQueue.readBuffer(minZBuffer);

		commandQueue.readBuffer(maxXBuffer);
		commandQueue.readBuffer(maxYBuffer);
		commandQueue.readBuffer(maxZBuffer);

		System.out.println();
		System.out.println("HOST");
		System.out.println(Arrays.toString(minX));
		System.out.println(Arrays.toString(minY));
		System.out.println(Arrays.toString(minZ));
		System.out.println(Arrays.toString(maxX));
		System.out.println(Arrays.toString(maxY));
		System.out.println(Arrays.toString(maxZ));
	}

	private static void generateBodies(float[] bodiesX, float[] bodiesY, float[] bodiesZ) {
		final Random random = new Random();
		for (int bodyIndex = 0; bodyIndex < NUMBER_OF_BODIES; ++bodyIndex) {
			bodiesX[bodyIndex] = (random.nextFloat() - 0.5f) * BODIES_RANGE;
			bodiesY[bodyIndex] = (random.nextFloat() - 0.5f) * BODIES_RANGE;
			bodiesZ[bodyIndex] = (random.nextFloat() - 0.5f) * BODIES_RANGE;
		}
	}

}
