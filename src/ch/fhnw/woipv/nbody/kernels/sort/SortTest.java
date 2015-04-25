package ch.fhnw.woipv.nbody.kernels.sort;

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

public class SortTest {

	private static final BuildOption DEBUG = new BuildOption("-D DEBUG2");

	private static final int WORK_GROUPS = 8;

	// TODO must be power of two?
	private static final int LOCAL_WORK_SIZE = 8;
	private static final int GLOBAL_WORK_SIZE = WORK_GROUPS * LOCAL_WORK_SIZE;

	// TODO must this be a multiple of global worksize?
	private static int NUMBER_OF_BODIES = 4;
	private static final float BODIES_RANGE = 1e3f;

	public static void main(String[] args) throws IOException {
		final BoundingBoxReduction boundingBoxReduction = new BoundingBoxReduction();
		final BuildTree buildTree = new BuildTree();
		final Sort summarizeTree = new Sort();

		final CLDevice device = CL20.createDevice();
		
		// TODO blocks are ComputeUnits
		// nnodes = nbodies * 2;
		// if (nnodes < 1024*blocks) nnodes = 1024*blocks;
		// while ((nnodes & (WARPSIZE-1)) != 0) nnodes++;
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
		final int bodyCount[] = new int[numberOfNodes + 1];
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
		commandQueue.readBuffer(bodiesXBuffer);

		System.out.println("child[]: " + Arrays.toString(child));
		System.out.println("bodiesX[]: " + Arrays.toString(bodiesX));

		int numNodes = bottom[0];

		buildTree.execute(context, commandQueue,
				bodiesXBuffer, bodiesYBuffer, bodiesZBuffer,
				blockCountBuffer, radiusBuffer, bottomBuffer, massBuffer, childBuffer, bodyCountBuffer,
				NUMBER_OF_BODIES, GLOBAL_WORK_SIZE, LOCAL_WORK_SIZE, WORK_GROUPS, numberOfNodes, warpSize, true);

		commandQueue.readBuffer(bodiesXBuffer);
		System.out.println("bodiesX[]: " + Arrays.toString(bodiesX));

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
		commandQueue.readBuffer(bodyCountBuffer);

		for (int i = 8 * (numberOfNodes + 1) - 1; i > 0; i -= 8) {
			System.out.print(numNodes-- + ": " + "[" + child[i - 7] + ", " + child[i - 6] + ", " + child[i - 5] + ", " + child[i - 4] + ", " + child[i - 3] + ", " + child[i - 2]
					+ ", " + child[i - 1] + ", " + child[i - 0] + "]");
			System.out.println();
		}

		System.out.println("bodyCount[]: " + Arrays.toString(bodyCount));
		System.out.println("mass[]: " + Arrays.toString(mass));
		System.out.println("bodiesX[]: " + Arrays.toString(bodiesX));
		System.out.println("bodiesY[]: " + Arrays.toString(bodiesY));
		System.out.println("bodiesZ[]: " + Arrays.toString(bodiesZ));

	}

	private static void generateRandomBodies(float[] bodiesX, float[] bodiesY, float[] bodiesZ, float[] mass) {
		// final Random random = new Random();
		// for (int bodyIndex = 0; bodyIndex < NUMBER_OF_BODIES; ++bodyIndex) {
		// bodiesX[bodyIndex] = (random.nextFloat() - 0.5f) * BODIES_RANGE;
		// bodiesY[bodyIndex] = (random.nextFloat() - 0.5f) * BODIES_RANGE;
		// bodiesZ[bodyIndex] = (random.nextFloat() - 0.5f) * BODIES_RANGE;
		// }
	}

	private static void generateBodies(float[] bodiesX, float[] bodiesY, float[] bodiesZ, float[] mass) {

		bodiesX[0] = 50; // 53 to check cell center of mass
		bodiesY[0] = 50;
		bodiesZ[0] = 50;
		mass[0] = 1;

		bodiesX[1] = -50;
		bodiesY[1] = 50;
		bodiesZ[1] = 50;
		mass[1] = 1;

		bodiesX[2] = 50;
		bodiesY[2] = -50;
		bodiesZ[2] = 50;
		mass[2] = 1;

		bodiesX[3] = -50;
		bodiesY[3] = -50;
		bodiesZ[3] = 50;
		mass[3] = 1;

		bodiesX[4] = 50;
		bodiesY[4] = 50;
		bodiesZ[4] = -50;
		mass[4] = 1;

		bodiesX[5] = -50;
		bodiesY[5] = 50;
		bodiesZ[5] = -50;
		mass[5] = 1;

		bodiesX[6] = 50;
		bodiesY[6] = -50;
		bodiesZ[6] = -50;
		mass[6] = 1;

		bodiesX[7] = -50;
		bodiesY[7] = -50;
		bodiesZ[7] = -50;
		mass[7] = 1;

		// ====

		bodiesX[8] = 5;
		bodiesY[8] = 5;
		bodiesZ[8] = 5;
		mass[8] = 1;

		bodiesX[9] = -5;
		bodiesY[9] = 5;
		bodiesZ[9] = 5;
		mass[9] = 1;

		bodiesX[10] = 5;
		bodiesY[10] = -5;
		bodiesZ[10] = 5;
		mass[10] = 1;

		bodiesX[11] = 5;
		bodiesY[11] = 5;
		bodiesZ[11] = -5;
		mass[11] = 1;

		bodiesX[12] = -5;
		bodiesY[12] = -5;
		bodiesZ[12] = 5;
		mass[12] = 1;

		bodiesX[13] = -5;
		bodiesY[13] = 5;
		bodiesZ[13] = -5;
		mass[13] = 1;

		bodiesX[14] = 5;
		bodiesY[14] = -5;
		bodiesZ[14] = -5;
		mass[14] = 1;

		bodiesX[15] = -5;
		bodiesY[15] = -5;
		bodiesZ[15] = -5;
		mass[15] = 1;

		NUMBER_OF_BODIES = 16;

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
