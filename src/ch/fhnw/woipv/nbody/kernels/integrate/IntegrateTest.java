package ch.fhnw.woipv.nbody.kernels.integrate;

import static org.jocl.CL.*;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import net.benjaminneukom.oocl.cl.CL20;
import net.benjaminneukom.oocl.cl.CLCommandQueue;
import net.benjaminneukom.oocl.cl.CLContext;
import net.benjaminneukom.oocl.cl.CLDevice;
import net.benjaminneukom.oocl.cl.CLMemory;
import ch.fhnw.woipv.nbody.kernels.boundsReduction.BoundingBoxReduction;
import ch.fhnw.woipv.nbody.kernels.buildTree.BuildTree;
import ch.fhnw.woipv.nbody.kernels.calculateForce.CalculateForce;
import ch.fhnw.woipv.nbody.kernels.sort.Sort;
import ch.fhnw.woipv.nbody.kernels.summarizeTree.SummarizeTree;

public class IntegrateTest {
	// TODO for cpus this needs to be one due to no lock stepping
	// TODO how to determine these values?
	private static final int WORK_GROUPS = 4; // THREADS (for now all the same)
	private static final int FACTORS = 1; // FACTORS (for now all the same)

	private static int bodies = 128;

	public static void main(String[] args) throws IOException {
		final BoundingBoxReduction boundingBoxReduction = new BoundingBoxReduction();
		final BuildTree buildTree = new BuildTree();
		final SummarizeTree summarizeTree = new SummarizeTree();
		final Sort sort = new Sort();
		final CalculateForce force = new CalculateForce();
		final Integrate integrate = new Integrate();

		final CLDevice device = CL20.createDevice();

		final int maxComputeUnits = (int)device.getLong(CL_DEVICE_MAX_COMPUTE_UNITS);
//		final int maxComputeUnits = 1;

		final int global = maxComputeUnits * WORK_GROUPS * FACTORS;
		final int local = WORK_GROUPS;

		int numberOfNodes = bodies * 2;
		final int warpSize = 64;
		if (numberOfNodes < 1024 * maxComputeUnits)
			numberOfNodes = 1024 * maxComputeUnits;
		while ((numberOfNodes & (warpSize - 1)) != 0)
			++numberOfNodes;

		final CLContext context = device.createContext();
		final CLCommandQueue commandQueue = context.createCommandQueue();

		final float bodiesX[] = new float[numberOfNodes + 1];
		final float bodiesY[] = new float[numberOfNodes + 1];
		final float bodiesZ[] = new float[numberOfNodes + 1];
		
		final float velX[] = new float[numberOfNodes + 1];
		final float velY[] = new float[numberOfNodes + 1];
		final float velZ[] = new float[numberOfNodes + 1];

		final float accX[] = new float[numberOfNodes + 1];
		final float accY[] = new float[numberOfNodes + 1];
		final float accZ[] = new float[numberOfNodes + 1];

		final int step[] = new int[] { -1 };
		final int blockCount[] = new int[] { 0 };
		final float radius[] = new float[1];
		final int bottom[] = new int[1];
		final float mass[] = new float[numberOfNodes + 1];
		final int bodyCount[] = new int[numberOfNodes + 1];
		final int child[] = new int[8 * (numberOfNodes + 1)];
		final int start[] = new int[numberOfNodes + 1];
		final int sorted[] = new int[numberOfNodes + 1];
		final int maxDepth[] = new int[] { 1 };

		generateRandomBodies(bodiesX, bodiesY, bodiesZ, mass);
		// generateBodies(bodiesX, bodiesY, bodiesZ, mass);

		final CLMemory bodiesXBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bodiesX);
		final CLMemory bodiesYBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bodiesY);
		final CLMemory bodiesZBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bodiesZ);
		
		final CLMemory velXBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, velX);
		final CLMemory velYBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, velY);
		final CLMemory velZBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, velZ);
		
		final CLMemory accXBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, accX);
		final CLMemory accYBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, accY);
		final CLMemory accZBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, accZ);

		final CLMemory stepBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, step);
		final CLMemory blockCountBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, blockCount);
		final CLMemory radiusBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, radius);
		final CLMemory maxDepthBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, maxDepth);
		final CLMemory bottomBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bottom);
		final CLMemory massBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mass);
		final CLMemory bodyCountBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bodyCount);

		final CLMemory childBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, child);
		final CLMemory startBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, start);
		final CLMemory sortedBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sorted);

		boundingBoxReduction.execute(context, commandQueue,
				bodiesXBuffer, bodiesYBuffer, bodiesZBuffer,
				velXBuffer, velYBuffer, velZBuffer,
				accXBuffer, accYBuffer, accZBuffer,
				stepBuffer, blockCountBuffer, radiusBuffer, maxDepthBuffer, bottomBuffer, massBuffer, childBuffer, bodyCountBuffer, startBuffer, sortedBuffer,
				bodies, global, local, WORK_GROUPS, numberOfNodes, warpSize, false);

		commandQueue.readBuffer(bottomBuffer);
		commandQueue.readBuffer(childBuffer);
		commandQueue.readBuffer(massBuffer);
		commandQueue.readBuffer(bodiesXBuffer);

		System.out.println("child[]: " + Arrays.toString(child));
		System.out.println("bodiesX[]: " + Arrays.toString(bodiesX));

		int numNodes = bottom[0];

		buildTree.execute(context, commandQueue,
				bodiesXBuffer, bodiesYBuffer, bodiesZBuffer,
				velXBuffer, velYBuffer, velZBuffer,
				accXBuffer, accYBuffer, accZBuffer,
				stepBuffer, blockCountBuffer, radiusBuffer, maxDepthBuffer, bottomBuffer, massBuffer, childBuffer, bodyCountBuffer, startBuffer, sortedBuffer,
				bodies, global, local, WORK_GROUPS, numberOfNodes, warpSize, false);

		commandQueue.readBuffer(bodiesXBuffer);
		commandQueue.readBuffer(bodiesYBuffer);
		commandQueue.readBuffer(bodiesZBuffer);
		commandQueue.readBuffer(childBuffer);

		System.out.println("bodiesX[]: " + Arrays.toString(bodiesX));
		System.out.println("bodiesY[]: " + Arrays.toString(bodiesY));
		System.out.println("bodiesZ[]: " + Arrays.toString(bodiesZ));

		int n = numNodes;
		for (int i = 8 * (numberOfNodes + 1) - 1; i > 0; i -= 8) {
			System.out.print(n-- + ": " + "[" + child[i - 7] + ", " + child[i - 6] + ", " + child[i - 5] + ", " + child[i - 4] + ", " + child[i - 3] + ", " + child[i - 2]
					+ ", " + child[i - 1] + ", " + child[i - 0] + "]");
			System.out.println();
		}

		summarizeTree.execute(context, commandQueue,
				bodiesXBuffer, bodiesYBuffer, bodiesZBuffer,
				velXBuffer, velYBuffer, velZBuffer,
				accXBuffer, accYBuffer, accZBuffer,
				stepBuffer, blockCountBuffer, radiusBuffer, maxDepthBuffer, bottomBuffer, massBuffer, childBuffer, bodyCountBuffer, startBuffer, sortedBuffer,
				bodies, global, local, WORK_GROUPS, numberOfNodes, warpSize, false);
		
		commandQueue.readBuffer(bodyCountBuffer);
		System.out.println("bodyCount[]: " + Arrays.toString(bodyCount));
		
		sort.execute(context, commandQueue,
				bodiesXBuffer, bodiesYBuffer, bodiesZBuffer,
				velXBuffer, velYBuffer, velZBuffer,
				accXBuffer, accYBuffer, accZBuffer,
				stepBuffer, blockCountBuffer, radiusBuffer, maxDepthBuffer, bottomBuffer, massBuffer, childBuffer, bodyCountBuffer, startBuffer, sortedBuffer,
				bodies, global, local, WORK_GROUPS, numberOfNodes, warpSize, false);

		force.execute(context, commandQueue,
				bodiesXBuffer, bodiesYBuffer, bodiesZBuffer,
				velXBuffer, velYBuffer, velZBuffer,
				accXBuffer, accYBuffer, accZBuffer,
				stepBuffer, blockCountBuffer, radiusBuffer, maxDepthBuffer, bottomBuffer, massBuffer, childBuffer, bodyCountBuffer, startBuffer, sortedBuffer,
				bodies, global, local, WORK_GROUPS, numberOfNodes, warpSize, false);
		
		integrate.execute(context, commandQueue,
				bodiesXBuffer, bodiesYBuffer, bodiesZBuffer,
				velXBuffer, velYBuffer, velZBuffer,
				accXBuffer, accYBuffer, accZBuffer,
				stepBuffer, blockCountBuffer, radiusBuffer, maxDepthBuffer, bottomBuffer, massBuffer, childBuffer, bodyCountBuffer, startBuffer, sortedBuffer,
				bodies, global, local, WORK_GROUPS, numberOfNodes, warpSize, false);

		
		commandQueue.readBuffer(bodiesXBuffer);
		commandQueue.readBuffer(bodiesYBuffer);
		commandQueue.readBuffer(bodiesZBuffer);
		commandQueue.readBuffer(accXBuffer);
		commandQueue.readBuffer(accYBuffer);
		commandQueue.readBuffer(accZBuffer);
		commandQueue.readBuffer(blockCountBuffer);
		commandQueue.readBuffer(radiusBuffer);
		commandQueue.readBuffer(bottomBuffer);
		commandQueue.readBuffer(massBuffer);
		commandQueue.readBuffer(childBuffer);
		commandQueue.readBuffer(bodyCountBuffer);
		commandQueue.readBuffer(sortedBuffer);
		commandQueue.readBuffer(startBuffer);

		n = numberOfNodes;
		for (int i = 8 * (numberOfNodes + 1) - 1; i > 0; i -= 8) {
			System.out.print(n-- + ": " + "[" + child[i - 7] + ", " + child[i - 6] + ", " + child[i - 5] + ", " + child[i - 4] + ", " + child[i - 3] + ", " + child[i - 2]
					+ ", " + child[i - 1] + ", " + child[i - 0] + "]");
			System.out.println();
		}

		System.out.println("bodyCount[]: " + Arrays.toString(bodyCount));
		System.out.println("mass[]: " + Arrays.toString(mass));
		System.out.println("sorted[]: " + Arrays.toString(sorted));
		System.out.println("start[]: " + Arrays.toString(start));
		System.out.println("bodiesX[]: " + Arrays.toString(bodiesX));
		System.out.println("bodiesY[]: " + Arrays.toString(bodiesY));
		System.out.println("bodiesZ[]: " + Arrays.toString(bodiesZ));
		System.out.println("accX[]: " + Arrays.toString(accX));
		System.out.println("accY[]: " + Arrays.toString(accY));
		System.out.println("accZ[]: " + Arrays.toString(accZ));

	}

	private static void generateRandomBodies(float[] bodiesX, float[] bodiesY, float[] bodiesZ, float[] mass) {
		final Random random = new Random();
		for (int bodyIndex = 0; bodyIndex < bodies; ++bodyIndex) {
			bodiesX[bodyIndex] = (float) ((random.nextFloat() - 0.5f) * 1e3);
			bodiesY[bodyIndex] = (float) ((random.nextFloat() - 0.5f) * 1e3);
			bodiesZ[bodyIndex] = (float) ((random.nextFloat() - 0.5f) * 1e3);
			mass[bodyIndex] = 1;
		}
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

		bodies = 16;

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
