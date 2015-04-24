package ch.fhnw.woipv.nbody.kernels.summarizeTree;

import static org.jocl.CL.*;

import java.io.IOException;
import java.util.Arrays;

import net.benjaminneukom.oocl.cl.CL20;
import net.benjaminneukom.oocl.cl.CLCommandQueue;
import net.benjaminneukom.oocl.cl.CLContext;
import net.benjaminneukom.oocl.cl.CLDevice;
import net.benjaminneukom.oocl.cl.CLMemory;
import net.benjaminneukom.oocl.cl.CLProgram.BuildOption;
import ch.fhnw.woipv.nbody.kernels.boundsReduction.BoundingBoxReduction;
import ch.fhnw.woipv.nbody.kernels.buildTree.BuildTree;

public class SummarizeTreeTest {

	private static final BuildOption DEBUG = new BuildOption("-D DEBUG2");

	private static final int WORK_GROUPS = 1;

	// TODO must be power of two?
	private static final int LOCAL_WORK_SIZE = 1;
	private static final int GLOBAL_WORK_SIZE = WORK_GROUPS * LOCAL_WORK_SIZE;

	// TODO must this be a multiple of global worksize?
	private static int NUMBER_OF_BODIES = 4;
	private static final float BODIES_RANGE = 1e3f;

	public static void main(String[] args) throws IOException {
		final BoundingBoxReduction boundingBoxReduction = new BoundingBoxReduction();
		final BuildTree buildTree = new BuildTree();
		final SummarizeTree summarizeTree = new SummarizeTree();

		final CLDevice device = CL20.createDevice();

		final int warpSize = 64;
		int numberOfNodes = NUMBER_OF_BODIES * 2;
		while ((numberOfNodes & (warpSize - 1)) != 0)
			++numberOfNodes;

		final CLContext context = device.createContext();
		final CLCommandQueue commandQueue = context.createCommandQueue();

		final float bodiesX[] = new float[numberOfNodes + 1];
		final float bodiesY[] = new float[numberOfNodes + 1];
		final float bodiesZ[] = new float[numberOfNodes + 1];

		final int blockCount[] = new int[1];
		final float radius[] = new float[1];
		final int bottom[] = new int[1];
		final float mass[] = new float[numberOfNodes + 1];
		final float bodyCount[] = new float[numberOfNodes + 1];
		final int child[] = new int[8 * (numberOfNodes + 1)];

		generateBodies(bodiesX, bodiesY, bodiesZ, mass);

		final CLMemory bodiesXBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bodiesX);
		final CLMemory bodiesYBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bodiesY);
		final CLMemory bodiesZBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bodiesZ);

		final CLMemory blockCountBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, blockCount);
		final CLMemory radiusBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, radius);
		final CLMemory bottomBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bottom);
		final CLMemory massBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mass);
		final CLMemory bodyCountBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bodyCount);

		final CLMemory childBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, child);

		boundingBoxReduction.execute(context, commandQueue,
				bodiesXBuffer, bodiesYBuffer, bodiesZBuffer,
				blockCountBuffer, radiusBuffer, bottomBuffer, massBuffer, childBuffer, bodyCountBuffer,
				NUMBER_OF_BODIES, GLOBAL_WORK_SIZE, LOCAL_WORK_SIZE, WORK_GROUPS, numberOfNodes, warpSize, true);

		commandQueue.readBuffer(bottomBuffer);
		commandQueue.readBuffer(childBuffer);
		commandQueue.readBuffer(massBuffer);
		commandQueue.finish();
		System.out.print("child[]: ");
		for (int i = 0; i < child.length; ++i) {
			System.out.print(child[i] + ", ");
		}
		System.out.println("]");

		int numNodes = bottom[0];

		buildTree.execute(context, commandQueue,
				bodiesXBuffer, bodiesYBuffer, bodiesZBuffer,
				blockCountBuffer,radiusBuffer, bottomBuffer, massBuffer, childBuffer, bodyCountBuffer,
				NUMBER_OF_BODIES, GLOBAL_WORK_SIZE, LOCAL_WORK_SIZE, WORK_GROUPS, numberOfNodes, warpSize, true);

		summarizeTree.execute(context, commandQueue,
				bodiesXBuffer, bodiesYBuffer, bodiesZBuffer,
				blockCountBuffer, radiusBuffer, bottomBuffer, massBuffer, childBuffer, bodyCountBuffer,
				NUMBER_OF_BODIES, GLOBAL_WORK_SIZE, LOCAL_WORK_SIZE, WORK_GROUPS, numberOfNodes, warpSize, true);

		commandQueue.readBuffer(bodiesXBuffer);
		commandQueue.readBuffer(bodiesYBuffer);
		commandQueue.readBuffer(bodiesZBuffer);
		commandQueue.readBuffer(blockCountBuffer);
		commandQueue.readBuffer(radiusBuffer);
		commandQueue.readBuffer(bottomBuffer);
		commandQueue.readBuffer(massBuffer);
		commandQueue.readBuffer(childBuffer);

		for (int i = 8 * (numberOfNodes + 1) - 1; i > 0; i -= 8) {
			System.out.print(numNodes-- + ": " + "[" + child[i - 7] + ", " + child[i - 6] + ", " + child[i - 5] + ", " + child[i - 4] + ", " + child[i - 3] + ", " + child[i - 2]
					+ ", " + child[i - 1] + ", " + child[i - 0] + "]");
			System.out.println();
		}
	}

	private static void generateBodies(float[] bodiesX, float[] bodiesY, float[] bodiesZ, float[] mass) {
		// final Random random = new Random();
		// for (int bodyIndex = 0; bodyIndex < NUMBER_OF_BODIES; ++bodyIndex) {
		// bodiesX[bodyIndex] = (random.nextFloat() - 0.5f) * BODIES_RANGE;
		// bodiesY[bodyIndex] = (random.nextFloat() - 0.5f) * BODIES_RANGE;
		// bodiesZ[bodyIndex] = (random.nextFloat() - 0.5f) * BODIES_RANGE;
		// }

		bodiesX[0] = 50;
		bodiesY[0] = 50;
		bodiesZ[0] = 50;
		mass[0] = 1;

		bodiesX[1] = -50;
		bodiesY[1] = 50;
		bodiesZ[1] = 50;
		mass[1] = 1;
		NUMBER_OF_BODIES = 2;

		//
		// bodiesX[2] = 50;
		// bodiesY[2] = -50;
		// bodiesZ[2] = 50;
		//
		// bodiesX[3] = -50;
		// bodiesY[3] = -50;
		// bodiesZ[3] = 50;
		//
		// bodiesX[4] = 50;
		// bodiesY[4] = 50;
		// bodiesZ[4] = -50;
		//
		// bodiesX[5] = -50;
		// bodiesY[5] = 50;
		// bodiesZ[5] = -50;
		//
		// bodiesX[6] = 50;
		// bodiesY[6] = -50;
		// bodiesZ[6] = -50;
		//
		// bodiesX[7] = -50;
		// bodiesY[7] = -50;
		// bodiesZ[7] = -50;

		// ====

		// bodiesX[8] = 45;
		// bodiesY[8] = 45;
		// bodiesZ[8] = -45;
		//
		// bodiesX[9] = 55;
		// bodiesY[9] = 55;
		// bodiesZ[9] = -55;
		//
		// bodiesX[10] = 56;
		// bodiesY[10] = 56;
		// bodiesZ[10] = -56;

		// NUMBER_OF_BODIES = 10;
	}

}
