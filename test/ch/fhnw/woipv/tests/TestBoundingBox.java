package ch.fhnw.woipv.tests;

import static org.jocl.CL.*;

import java.io.File;
import java.util.Arrays;

import static org.junit.Assert.*;
import net.benjaminneukom.oocl.cl.CLCommandQueue;
import net.benjaminneukom.oocl.cl.CLContext;
import net.benjaminneukom.oocl.cl.CLDevice;
import net.benjaminneukom.oocl.cl.CLDevice.DeviceType;
import net.benjaminneukom.oocl.cl.CLKernel;
import net.benjaminneukom.oocl.cl.CLMemory;
import net.benjaminneukom.oocl.cl.CLPlatform;
import net.benjaminneukom.oocl.cl.CLProgram.BuildOption;

import org.jocl.CL;
import org.junit.Before;
import org.junit.Test;

import ch.fhnw.woipv.nbody.simulation.universe.RandomCubicUniverseGenerator;
import ch.fhnw.woipv.nbody.simulation.universe.UniverseGenerator;

public class TestBoundingBox {

	static {
		CL.setExceptionsEnabled(true);
	}

	private static final float FLOAT_MAX = 1_000_000_000;
	private static final float EPSILON = 0.00001f;

	private static final boolean HOST_DEBUG = true;
	private static final boolean KERNEL_DEBUG = false;

	private static final int WARPSIZE = 64;
	private static final int WORK_GROUPS = 4; // THREADS (for now all the same)
	private static final int FACTORS = 1; // FACTORS (for now all the same)

	private BuildOption[] buildOptions;
	private int numberOfNodes;
	private int nbodies;

	private int maxComputeUnits;
	private int global;
	private int local;

	private CLContext context;
	private CLDevice device;
	private CLCommandQueue commandQueue;

	private CLKernel boundingBoxKernel;
	private CLKernel buildTreeKernel;
	private CLKernel calculateForceKernel;
	private CLKernel sortKernel;
	private CLKernel summarizeKernel;
	private CLKernel integrateKernel;

	private CLKernel[] simulationKernels;

	private CLMemory<float[]> bodiesXBuffer;
	private CLMemory<float[]> bodiesYBuffer;
	private CLMemory<float[]> bodiesZBuffer;

	private CLMemory<float[]> velXBuffer;
	private CLMemory<float[]> velYBuffer;
	private CLMemory<float[]> velZBuffer;

	private CLMemory<float[]> accXBuffer;
	private CLMemory<float[]> accYBuffer;
	private CLMemory<float[]> accZBuffer;

	private CLMemory<int[]> stepBuffer;
	private CLMemory<int[]> blockCountBuffer;
	private CLMemory<float[]> radiusBuffer;
	private CLMemory<int[]> maxDepthBuffer;
	private CLMemory<int[]> bottomBuffer;
	private CLMemory<float[]> massBuffer;
	private CLMemory<int[]> bodyCountBuffer;

	private CLMemory<int[]> childBuffer;
	private CLMemory<int[]> startBuffer;
	private CLMemory<int[]> sortedBuffer;

	private CLMemory<int[]> errorBuffer;

	private UniverseGenerator universeGenerator;

	@Before
	public void init() {

		this.nbodies = 2048;
		this.universeGenerator = new RandomCubicUniverseGenerator(nbodies);

		// init opencl
		this.device = CLPlatform.getFirst().getDevice(DeviceType.GPU, d -> d.getDeviceVersion() >= 2.0f).orElseThrow(() -> new IllegalStateException());
		this.context = device.createContext();
		this.commandQueue = this.context.createCommandQueue();

		// calculate workloads
//		this.maxComputeUnits = (int) device.getLong(CL_DEVICE_MAX_COMPUTE_UNITS);
		this.maxComputeUnits = 1;

		this.global = maxComputeUnits * WORK_GROUPS * FACTORS;
		this.local = WORK_GROUPS;
		this.numberOfNodes = calculateNumberOfNodes(nbodies, maxComputeUnits);

		this.buildOptions = createBuildOptions(numberOfNodes, nbodies, local, WORK_GROUPS);

		final float[] bodiesX = new float[numberOfNodes + 1];
		final float[] bodiesY = new float[numberOfNodes + 1];
		final float[] bodiesZ = new float[numberOfNodes + 1];

		final float[] velX = new float[numberOfNodes + 1];
		final float[] velY = new float[numberOfNodes + 1];
		final float[] velZ = new float[numberOfNodes + 1];

		final float[] bodiesMass = new float[numberOfNodes + 1];

		universeGenerator.generate(0, nbodies, bodiesX, bodiesY, bodiesZ, velX, velY, velZ, bodiesMass);

		loadBuffers(numberOfNodes, nbodies, bodiesX, bodiesY, bodiesZ, velX, velY, velZ, bodiesMass);
		loadKernels(context, buildOptions);

		setSimulationKernelsArguments(simulationKernels);
	}

	private void loadBuffers(final int numberOfNodes, final int nbodies, final float[] bodiesX, final float[] bodiesY, final float[] bodiesZ, final float[] velX,
			final float[] velY, final float[] velZ, final float[] mass) {
		this.bodiesXBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bodiesX);
		this.bodiesYBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bodiesY);
		this.bodiesZBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bodiesZ);

		this.velXBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, velX);
		this.velYBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, velY);
		this.velZBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, velZ);

		this.accXBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, new float[numberOfNodes + 1]);
		this.accYBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, new float[numberOfNodes + 1]);
		this.accZBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, new float[numberOfNodes + 1]);

		this.stepBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, new int[] { -1 });
		this.blockCountBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, new int[] { 0 });
		this.radiusBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, new float[1]);
		this.maxDepthBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, new int[] { 1 });
		this.bottomBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, new int[1]);
		this.massBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mass);
		this.bodyCountBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, new int[numberOfNodes + 1]);

		this.childBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, new int[8 * (numberOfNodes + 1)]);
		this.startBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, new int[numberOfNodes + 1]);
		this.sortedBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, new int[numberOfNodes + 1]);

		this.errorBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, new int[1]);

	}

	private void loadKernels(final CLContext context, final BuildOption[] options) {
		this.boundingBoxKernel = context.createKernel(new File("kernels/nbody/boundingbox.cl"), "boundingBox", options);
		this.buildTreeKernel = context.createKernel(new File("kernels/nbody/buildtree.cl"), "buildTree", options);
		this.summarizeKernel = context.createKernel(new File("kernels/nbody/summarizetree.cl"), "summarizeTree", options);
		this.sortKernel = context.createKernel(new File("kernels/nbody/sort.cl"), "sort", options);
		this.calculateForceKernel = context.createKernel(new File("kernels/nbody/calculateforce.cl"), "calculateForce", options);
		this.integrateKernel = context.createKernel(new File("kernels/nbody/integrate.cl"), "integrate", options);

		this.simulationKernels = new CLKernel[] { boundingBoxKernel, buildTreeKernel, summarizeKernel, sortKernel, calculateForceKernel, integrateKernel };
	}

	private void setSimulationKernelsArguments(final CLKernel[] kernels) {
		Arrays.stream(kernels).forEach(kernel -> setSimulationKernelArguments(kernel));
	}

	private void setSimulationKernelArguments(final CLKernel kernel) {
		kernel.setArguments(
				bodiesXBuffer, bodiesYBuffer, bodiesZBuffer,
				velXBuffer, velYBuffer, velZBuffer,
				accXBuffer, accYBuffer, accZBuffer, stepBuffer, blockCountBuffer,
				bodyCountBuffer, radiusBuffer, maxDepthBuffer, bottomBuffer, massBuffer, childBuffer, startBuffer, sortedBuffer, errorBuffer);

	}

	private static BuildOption[] createBuildOptions(final int numberOfNodes, final int nbodies, final int localWorkSize, final int numWorkGroups) {
		return new BuildOption[] {
				BuildOption.CL20,
				BuildOption.MAD,
				new BuildOption("-D NUMBER_OF_NODES=" + numberOfNodes),
				new BuildOption("-D NBODIES=" + nbodies),
				new BuildOption("-D WORKGROUP_SIZE=" + localWorkSize),
				new BuildOption("-D NUM_WORK_GROUPS=" + numWorkGroups),
				new BuildOption("-D DEBUG")
		};
	}

	private static int calculateNumberOfNodes(final int nbodies, final int maxComputeUnits) {
		int numberOfNodes = nbodies * 2;
		if (numberOfNodes < 1024 * maxComputeUnits)
			numberOfNodes = 1024 * maxComputeUnits;
		while ((numberOfNodes & (WARPSIZE - 1)) != 0)
			++numberOfNodes;

		return numberOfNodes;
	}

	@Test
	public void testRoot() {
		executeSimulationKernel(boundingBoxKernel);

		commandQueue.finish();

		float minX = FLOAT_MAX, minY = FLOAT_MAX, minZ = FLOAT_MAX;
		float maxX = -FLOAT_MAX, maxY = -FLOAT_MAX, maxZ = -FLOAT_MAX;

		commandQueue.readBuffer(bodiesXBuffer);
		commandQueue.readBuffer(bodiesYBuffer);
		commandQueue.readBuffer(bodiesZBuffer);

		float[] bodyX = bodiesXBuffer.getData();
		float[] bodyY = bodiesYBuffer.getData();
		float[] bodyZ = bodiesZBuffer.getData();

		for (int i = 0; i < nbodies; ++i) {
			minX = Math.min(minX, bodyX[i]);
			minY = Math.min(minY, bodyY[i]);
			minZ = Math.min(minZ, bodyZ[i]);

			maxX = Math.max(maxX, bodyX[i]);
			maxY = Math.max(maxY, bodyY[i]);
			maxZ = Math.max(maxZ, bodyZ[i]);
		}

		System.out.println(Arrays.toString(bodyX));
		System.out.println(Arrays.toString(bodyY));
		System.out.println(Arrays.toString(bodyZ));

		float rootX = 0.5f * (minX + maxX);
		float rootY = 0.5f * (minY + maxY);
		float rootZ = 0.5f * (minZ + maxZ);

		float clRootX = bodyX[numberOfNodes];
		float clRootY = bodyY[numberOfNodes];
		float clRootZ = bodyZ[numberOfNodes];

		assertTrue(epsilonEquals(rootX, clRootX));
		assertTrue(epsilonEquals(rootY, clRootY));
		assertTrue(epsilonEquals(rootZ, clRootZ));
	}

	private static boolean epsilonEquals(float a, float b) {
		return Math.abs(a - b) < EPSILON;
	}

	private void executeSimulationKernel(CLKernel kernel) {
		commandQueue.execute(kernel, 1, global, local);
		if (HOST_DEBUG) {
			commandQueue.readBuffer(errorBuffer);
			if (errorBuffer.getData()[0] != 0) {
				System.out.println("===");

				System.exit(0);
			}
		}
	}

}
