package ch.fhnw.woipv.nbody;

import static org.jocl.CL.*;

import java.io.File;
import java.util.Arrays;

import org.jocl.CL;

import ch.fhnw.woipv.nbody.simulation.NBodySimulation;
import ch.fhnw.woipv.nbody.simulation.generators.RandomUniverseGenerator;
import ch.fhnw.woipv.nbody.simulation.generators.UniverseGenerator;
import net.benjaminneukom.oocl.cl.CLCommandQueue;
import net.benjaminneukom.oocl.cl.CLContext;
import net.benjaminneukom.oocl.cl.CLDevice;
import net.benjaminneukom.oocl.cl.CLMemory;
import net.benjaminneukom.oocl.cl.CLDevice.DeviceType;
import net.benjaminneukom.oocl.cl.CLKernel;
import net.benjaminneukom.oocl.cl.CLPlatform;
import net.benjaminneukom.oocl.cl.CLProgram.BuildOption;

public class GpuNBodySimulation implements NBodySimulation {

	static {
		CL.setExceptionsEnabled(true);
	}

	private static final int WARPSIZE = 64;
	private static final int WORK_GROUPS = 4; // THREADS (for now all the same)
	private static final int FACTORS = 1; // FACTORS (for now all the same)

	private final BuildOption[] buildOptions;
	private final int numberOfNodes;
	private final int nbodies;

	private final int maxComputeUnits;
	private final int global;
	private final int local;

	private CLContext context;
	private CLDevice device;
	private CLCommandQueue commandQueue;

	private CLKernel boundingBoxKernel;
	private CLKernel buildTreeKernel;
	private CLKernel calculateForceKernel;
	private CLKernel sortKernel;
	private CLKernel summarizeKernel;
	private CLKernel integrateKernel;

	private CLKernel[] kernels;

	private CLMemory bodiesXBuffer;
	private CLMemory bodiesYBuffer;
	private CLMemory bodiesZBuffer;

	private CLMemory velXBuffer;
	private CLMemory velYBuffer;
	private CLMemory velZBuffer;

	private CLMemory accXBuffer;
	private CLMemory accYBuffer;
	private CLMemory accZBuffer;

	private CLMemory stepBuffer;
	private CLMemory blockCountBuffer;
	private CLMemory radiusBuffer;
	private CLMemory maxDepthBuffer;
	private CLMemory bottomBuffer;
	private CLMemory massBuffer;
	private CLMemory bodyCountBuffer;

	private CLMemory childBuffer;
	private CLMemory startBuffer;
	private CLMemory sortedBuffer;
	
	private float[] bodiesXArr;
	private float[] bodiesYArr;
	private float[] bodiesZArr;
	private float[] bodiesMass;

	public GpuNBodySimulation(final int nbodies, UniverseGenerator generator) {

		this.device = CLPlatform.getFirst().getDevice(DeviceType.GPU, d -> d.getDeviceVersion() >= 2.0f).orElseThrow(() -> new IllegalStateException());
		this.context = device.createContext();
		this.commandQueue = this.context.createCommandQueue();

		this.maxComputeUnits = (int) device.getLong(CL_DEVICE_MAX_COMPUTE_UNITS);
//		this.maxComputeUnits = 1;

		this.global = maxComputeUnits * WORK_GROUPS * FACTORS;
		this.local = WORK_GROUPS;
		this.nbodies = nbodies;
		this.numberOfNodes = calculateNumberOfNodes(nbodies, maxComputeUnits);

		this.buildOptions = createBuildOptions(numberOfNodes, nbodies, local, WORK_GROUPS);

		bodiesXArr = new float[numberOfNodes + 1];
		bodiesYArr = new float[numberOfNodes + 1];
		bodiesZArr = new float[numberOfNodes + 1];
		
		bodiesMass = new float[numberOfNodes + 1];

		generator.generate(0, nbodies, bodiesXArr, bodiesYArr, bodiesZArr, bodiesMass);

		loadBuffers(numberOfNodes, bodiesXArr, bodiesYArr, bodiesZArr, bodiesMass);
		loadKernels(context, buildOptions);

		setKernelArguments(kernels);
	}

	private void loadBuffers(int numberOfNodes, float[] bodiesX, float[] bodiesY, float[] bodiesZ, float[] mass) {
		this.bodiesXBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bodiesX);
		this.bodiesYBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bodiesY);
		this.bodiesZBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bodiesZ);

		this.velXBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, new float[numberOfNodes + 1]);
		this.velYBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, new float[numberOfNodes + 1]);
		this.velZBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, new float[numberOfNodes + 1]);

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
	}

	private void loadKernels(CLContext context, BuildOption[] options) {
		this.boundingBoxKernel = context.createKernel(new File("kernels/nbody/boundingbox.cl"), "boundingBox", options);
		this.buildTreeKernel = context.createKernel(new File("kernels/nbody/buildtree.cl"), "buildTree", options);
		this.summarizeKernel = context.createKernel(new File("kernels/nbody/summarizetree.cl"), "summarizeTree", options);
		this.sortKernel = context.createKernel(new File("kernels/nbody/sort.cl"), "sort", options);
		this.calculateForceKernel = context.createKernel(new File("kernels/nbody/calculateforce.cl"), "calculateForce", options);
		this.integrateKernel = context.createKernel(new File("kernels/nbody/integrate.cl"), "integrate", options);

		this.kernels = new CLKernel[] { boundingBoxKernel, buildTreeKernel, summarizeKernel, sortKernel, calculateForceKernel, integrateKernel };
	}

	private void setKernelArguments(CLKernel[] kernels) {
		Arrays.stream(kernels).forEach(kernel -> setArguments(kernel));
	}

	private void setArguments(CLKernel kernel) {
		kernel.setArguments(	
				bodiesXBuffer, bodiesYBuffer, bodiesZBuffer,
				velXBuffer, velYBuffer, velZBuffer,
				accXBuffer, accYBuffer, accZBuffer,
				stepBuffer, blockCountBuffer, bodyCountBuffer, radiusBuffer, maxDepthBuffer, bottomBuffer, massBuffer, childBuffer,startBuffer, sortedBuffer
			);
	}

	private static BuildOption[] createBuildOptions(int numberOfNodes, int nbodies, int localWorkSize, int numWorkGroups) {
		return new BuildOption[] {
				BuildOption.CL20,
				BuildOption.MAD,
				new BuildOption("-D NUMBER_OF_NODES=" + numberOfNodes),
				new BuildOption("-D NBODIES=" + nbodies),
				new BuildOption("-D WORKGROUP_SIZE=" + localWorkSize),
				new BuildOption("-D NUM_WORK_GROUPS=" + numWorkGroups),
//				new BuildOption("-D DEBUG")
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

	public void step() {
		System.out.println("step");
		commandQueue.execute(boundingBoxKernel, 1, global, local);
		commandQueue.execute(buildTreeKernel, 1, global, local);
		commandQueue.execute(summarizeKernel, 1, global, local);
		commandQueue.execute(sortKernel, 1, global, local);
		commandQueue.execute(calculateForceKernel, 1, global, local);
		commandQueue.execute(integrateKernel, 1, global, local);
		
		commandQueue.flush();
		commandQueue.finish();

		commandQueue.readBuffer(bodiesXBuffer);
		commandQueue.readBuffer(bodiesYBuffer);
		commandQueue.readBuffer(bodiesZBuffer);
		
		System.out.println("bodiesX: " + Arrays.toString(bodiesXArr));
		System.out.println("bodiesY: " + Arrays.toString(bodiesYArr));
		System.out.println("bodiesZ: " + Arrays.toString(bodiesZArr));
		System.out.println("======");
	}

	public static void main(String[] args) {
		final int nbodies = 8;
		final  GpuNBodySimulation nBodySimulation = new GpuNBodySimulation(nbodies, new RandomUniverseGenerator(2));

		for (int i = 0; i < 10; ++i) {
			nBodySimulation.step();
		}
	}

}
