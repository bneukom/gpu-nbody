package ch.fhnw.woipv.nbody.kernels.buildTree;

import static org.jocl.CL.*;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import ch.fhnw.woipv.nbody.internal.opencl.CL20;
import ch.fhnw.woipv.nbody.internal.opencl.CLCommandQueue;
import ch.fhnw.woipv.nbody.internal.opencl.CLContext;
import ch.fhnw.woipv.nbody.internal.opencl.CLDevice;
import ch.fhnw.woipv.nbody.internal.opencl.CLMemory;
import ch.fhnw.woipv.nbody.internal.opencl.CLProgram.BuildOption;
import ch.fhnw.woipv.nbody.kernels.boundsReduction.BoundingBoxReduction;

public class BuildTreeTest {

	private static final BuildOption DEBUG = new BuildOption("-D DEBUG2");

	private static final int WORK_GROUPS = 512;

	// TODO must be power of two?
	private static final int LOCAL_WORK_SIZE = 16;
	private static final int GLOBAL_WORK_SIZE = WORK_GROUPS * LOCAL_WORK_SIZE;

	// TODO must this be a multiple of global worksize?
	private static final int NUMBER_OF_BODIES = 2048 * 1024;
	private static final float BODIES_RANGE = (float) 1e9; 

	public static void main(String[] args) throws IOException {
		final BoundingBoxReduction boundingBoxReduction = new BoundingBoxReduction();
		final BuildTree buildTree = new BuildTree();

		final CLDevice device = CL20.createDevice();

		final int waveFrontSize = 64;
		int numberOfNodes = NUMBER_OF_BODIES * 2;
		while ((numberOfNodes & (waveFrontSize - 1)) != 0)
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
		final int child[] = new int[8 * (numberOfNodes + 1)];

		generateBodies(bodiesX, bodiesY, bodiesZ);

		final CLMemory bodiesXBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bodiesX);
		final CLMemory bodiesYBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bodiesY);
		final CLMemory bodiesZBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bodiesZ);

		final CLMemory blockCountBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, blockCount);
		final CLMemory radiusBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, radius);
		final CLMemory bottomBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bottom);
		final CLMemory massBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mass);
		final CLMemory childBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, child);

		boundingBoxReduction.calculateBoundingBox(context, commandQueue,
				bodiesXBuffer, bodiesYBuffer, bodiesZBuffer,
				blockCountBuffer, radiusBuffer, bottomBuffer, massBuffer, childBuffer,
				NUMBER_OF_BODIES, GLOBAL_WORK_SIZE, LOCAL_WORK_SIZE, WORK_GROUPS, numberOfNodes);

		commandQueue.readBuffer(bottomBuffer);
		int numNodes = bottom[0];

		buildTree.buildTree(context, commandQueue,
				bodiesXBuffer, bodiesYBuffer, bodiesZBuffer,
				blockCountBuffer, radiusBuffer, bottomBuffer, massBuffer, childBuffer,
				NUMBER_OF_BODIES, GLOBAL_WORK_SIZE, LOCAL_WORK_SIZE, WORK_GROUPS, numberOfNodes);

		commandQueue.readBuffer(bodiesXBuffer);
		commandQueue.readBuffer(bodiesYBuffer);
		commandQueue.readBuffer(bodiesZBuffer);
		commandQueue.readBuffer(blockCountBuffer);
		commandQueue.readBuffer(radiusBuffer);
		commandQueue.readBuffer(bottomBuffer);
		commandQueue.readBuffer(massBuffer);
		commandQueue.readBuffer(childBuffer);

		for (int i = 8 * (numberOfNodes + 1) - 1; i > NUMBER_OF_BODIES; i -= 8) {
			System.out.print(numNodes-- + ": " + "[" + child[i - 7] + ", " + child[i - 6] + ", " + child[i - 5] + ", " + child[i - 4] + ", " + child[i - 3] + ", " + child[i - 2]
					+ ", " + child[i - 1] + ", " + child[i - 0] + "]");
			System.out.println();
		}
		System.out.println("child[]: " + Arrays.toString(child));
		System.out.println("mass[]: " + Arrays.toString(mass));
		
		System.out.print("bodies[]: ");
		for (int i = 0; i < numberOfNodes + 1; ++i) {
			System.out.print("(" + bodiesX[i] + ", " + bodiesY[i] + ", " + bodiesZ[i] + "), ");
		}
		System.out.println();
	}

	private static void generateBodies(float[] bodiesX, float[] bodiesY, float[] bodiesZ) {
		final Random random = new Random();
		for (int bodyIndex = 0; bodyIndex < NUMBER_OF_BODIES; ++bodyIndex) {
			bodiesX[bodyIndex] = (random.nextFloat() - 0.5f) * BODIES_RANGE;
			bodiesY[bodyIndex] = (random.nextFloat() - 0.5f) * BODIES_RANGE;
			bodiesZ[bodyIndex] = (random.nextFloat() - 0.5f) * BODIES_RANGE;
		}

		// bodiesX[0] = 15.00001f;
		// bodiesY[0] = 0;
		// bodiesZ[0] = 0;
		// bodiesX[1] = 15;
		// bodiesY[1] = 0;
		// bodiesZ[1] = 0;
		//
		// bodiesX[2] = 16;
		// bodiesY[2] = 0;
		// bodiesZ[2] = 0;
		// bodiesX[3] = 16.001f;
		// bodiesY[3] = 0;
		// bodiesZ[3] = 0;
		// NUMBER_OF_BODIES = 4;

		// bodiesX[0] = 50;
		// bodiesY[0] = 50;
		// bodiesZ[0] = 50;
		//
		// bodiesX[1] = -50;
		// bodiesY[1] = 50;
		// bodiesZ[1] = 50;
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
		//
		// // ====
		//
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
		//
		// NUMBER_OF_BODIES = 10;
	}

}
