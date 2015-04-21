package ch.fhnw.woipv.nbody.kernels.boundsReduction;

import static org.jocl.CL.*;

import java.io.IOException;
import java.util.Random;

import ch.fhnw.woipv.nbody.internal.opencl.CL20;
import ch.fhnw.woipv.nbody.internal.opencl.CLCommandQueue;
import ch.fhnw.woipv.nbody.internal.opencl.CLContext;
import ch.fhnw.woipv.nbody.internal.opencl.CLDevice;
import ch.fhnw.woipv.nbody.internal.opencl.CLMemory;
import ch.fhnw.woipv.nbody.internal.opencl.CLProgram.BuildOption;

public class BoundsReductionTest {

	private static final BuildOption DEBUG = new BuildOption("-D DEBUG2");

	private static final int WORK_GROUPS = 4;

	// TODO must be power of two?
	private static final int LOCAL_WORK_SIZE = 8;
	private static final int GLOBAL_WORK_SIZE = WORK_GROUPS * LOCAL_WORK_SIZE;

	// TODO must this be a multiple of global worksize?
	private static final int NUMBER_OF_BODIES = GLOBAL_WORK_SIZE;
	private static final float BODIES_RANGE = 1e3f;

	public static void main(String[] args) throws IOException {
		final BoundingBoxReduction boundingBoxReduction = new BoundingBoxReduction();
		final CLDevice device = CL20.createDevice();
		
//		CL_DEVICE_WAVEFRONT_WIDTH_AMD
		
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

		float hostMinX = Float.MAX_VALUE;
		for (int i = 0; i < NUMBER_OF_BODIES; ++i) {
			hostMinX = Math.min(hostMinX, bodiesX[i]);
		}

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

		commandQueue.readBuffer(bodiesXBuffer);
		commandQueue.readBuffer(bodiesYBuffer);
		commandQueue.readBuffer(bodiesZBuffer);
		commandQueue.readBuffer(blockCountBuffer);
		commandQueue.readBuffer(radiusBuffer);
		commandQueue.readBuffer(bottomBuffer);
		commandQueue.readBuffer(massBuffer);
		commandQueue.readBuffer(childBuffer);
		
		System.out.println("Radius: " + radius[0]);
		System.out.println("BlockCount: " + blockCount[0]);
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
