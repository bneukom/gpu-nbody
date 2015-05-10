package ch.fhnw.woipv.nbody.simulation;

import static com.jogamp.opengl.GL.*;
import static org.jocl.CL.*;

import java.io.File;
import java.util.Arrays;

import jogamp.opengl.GLContextImpl;
import jogamp.opengl.GLDrawableImpl;
import jogamp.opengl.egl.EGLContext;
import jogamp.opengl.macosx.cgl.MacOSXCGLContext;
import jogamp.opengl.windows.wgl.WindowsWGLContext;
import jogamp.opengl.x11.glx.X11GLXContext;
import net.benjaminneukom.oocl.cl.CLCommandQueue;
import net.benjaminneukom.oocl.cl.CLContext;
import net.benjaminneukom.oocl.cl.CLDevice;
import net.benjaminneukom.oocl.cl.CLDevice.DeviceType;
import net.benjaminneukom.oocl.cl.CLKernel;
import net.benjaminneukom.oocl.cl.CLMemory;
import net.benjaminneukom.oocl.cl.CLPlatform;
import net.benjaminneukom.oocl.cl.CLProgram.BuildOption;

import org.jocl.CL;
import org.jocl.Sizeof;
import org.jocl.cl_context_properties;

import ch.fhnw.woipv.nbody.simulation.universe.PlummerUniverseGenerator;
import ch.fhnw.woipv.nbody.simulation.universe.UniverseGenerator;

import com.jogamp.nativewindow.NativeSurface;
import com.jogamp.opengl.GL;
import com.jogamp.opengl.GL3;
import com.jogamp.opengl.GLContext;

// TODO cleaner init
// TODO mode with and without opengl?
public class GpuNBodySimulation implements NBodySimulation {

	static {
		CL.setExceptionsEnabled(true);
	}

	public enum Mode {
		GL_INTEROP, DEFAULT
	}

	private final Mode mode;

	private static final boolean HOST_DEBUG = false;
	private static final boolean KERNEL_DEBUG = false;

	private static final int WARPSIZE = 64;
	private static final int WORK_GROUPS = 4; // THREADS (for now all the same)
	private static final int FACTORS = 1; // FACTORS (for now all the same)

	private BuildOption[] buildOptions;
	private int numberOfNodes;
	private final int nbodies;

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
	private CLKernel copyVertices;

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

	private CLMemory<Void> positionBuffer;

	private final UniverseGenerator universeGenerator;

	public GpuNBodySimulation(final Mode mode, final int nbodies, final UniverseGenerator generator) {
		this.mode = mode;
		this.nbodies = nbodies;
		this.universeGenerator = generator;
	}

	@Override
	public void init(final GL3 gl) {

		// Initialize the context properties
		final cl_context_properties contextProperties = new cl_context_properties();
		if (gl != null) {
			initContextProperties(contextProperties, gl);
		}

		// init opencl
		this.device = CLPlatform.getFirst().getDevice(DeviceType.GPU, d -> d.getDeviceVersion() >= 2.0f).orElseThrow(() -> new IllegalStateException());
		this.context = device.createContext(contextProperties);
		this.commandQueue = this.context.createCommandQueue();

		// calculate workloads
		// this.maxComputeUnits = (int) device.getLong(CL_DEVICE_MAX_COMPUTE_UNITS);
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
		this.copyVertices = context.createKernel(new File("kernels/nbody/copyvertices.cl"), "copyVertices", options);

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
				KERNEL_DEBUG ? new BuildOption("-D DEBUG") : BuildOption.EMPTY
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

	@Override
	public void initPositionBuffer(final GL3 gl, final int vbo) {
		if (positionBuffer != null) {
			positionBuffer.release();
		}

		if (gl != null) {
			gl.glBindBuffer(GL_ARRAY_BUFFER, vbo);
			positionBuffer = context.createFromGLBuffer(CL_MEM_WRITE_ONLY, vbo);
		} else {
			int size = nbodies * 4 * Sizeof.cl_float;
			positionBuffer = context.createEmptyBuffer(CL_MEM_WRITE_ONLY, size);
		}

		copyVertices.setArguments(bodiesXBuffer, bodiesYBuffer, bodiesZBuffer, positionBuffer);
	}

	@Override
	public void step() {
		if (HOST_DEBUG)
			System.out.println("step");

		if (mode == Mode.GL_INTEROP)
			commandQueue.enqueAcquireGLObject(positionBuffer);

		executeSimulationKernel(boundingBoxKernel);
		executeSimulationKernel(buildTreeKernel);
		executeSimulationKernel(summarizeKernel);
		executeSimulationKernel(sortKernel);
		executeSimulationKernel(calculateForceKernel);
		executeSimulationKernel(integrateKernel);

		if (mode == Mode.GL_INTEROP)
			commandQueue.execute(copyVertices, 1, global, local);

		commandQueue.finish();
	}

	private void executeSimulationKernel(CLKernel kernel) {
		commandQueue.execute(kernel, 1, global, local);
		if (HOST_DEBUG) {
			commandQueue.readBuffer(errorBuffer);
			if (errorBuffer.getData()[0] != 0) {
				System.out.println("exit");
				System.out.println("===");

				printChildren();

				System.out.println("===");

				printPosition();

				System.exit(0);
			}
		}
	}

	private void printChildren() {
		commandQueue.readBuffer(childBuffer);
		int[] child = childBuffer.getData();

		int n = numberOfNodes;
		for (int i = 8 * (numberOfNodes + 1) - 1; i > 0; i -= 8) {
			System.out.print(n-- + ": " + "[" + child[i - 7] + ", " + child[i - 6] + ", " + child[i - 5] + ", " + child[i - 4] + ", " + child[i - 3] + ", " + child[i - 2]
					+ ", " + child[i - 1] + ", " + child[i - 0] + "]");
			System.out.println();
		}
	}

	public void printEnergy() {
		commandQueue.readBuffer(massBuffer);
		commandQueue.readBuffer(velXBuffer);
		commandQueue.readBuffer(velYBuffer);
		commandQueue.readBuffer(velZBuffer);

		// TODO calculate gravitational energy.
		double totalEnergy = 0.0f;
		final float[] mass = massBuffer.getData();
		final float[] velX = velXBuffer.getData();
		final float[] velY = velYBuffer.getData();
		final float[] velZ = velZBuffer.getData();
		for (int i = 0; i < nbodies; ++i) {
			final double velocity = Math.sqrt(velX[i] * velX[i] + velY[i] * velY[i] + velZ[i] * velZ[i]);
			totalEnergy += 0.5 * mass[i] * velocity * velocity;
		}

		System.out.println("total energy: " + totalEnergy);
	}

	public void printTotalForce() {
		commandQueue.readBuffer(massBuffer);
		commandQueue.readBuffer(accXBuffer);
		commandQueue.readBuffer(accYBuffer);
		commandQueue.readBuffer(accZBuffer);

		float fx = 0;
		float fy = 0;
		float fz = 0;

		final float[] mass = massBuffer.getData();
		final float[] accX = accXBuffer.getData();
		final float[] accY = accYBuffer.getData();
		final float[] accZ = accZBuffer.getData();
		for (int i = 0; i < nbodies; ++i) {
			final float m = mass[i];
			fx += m * accX[i];
			fy += m * accY[i];
			fz += m * accZ[i];
		}

		System.out.println("total force: (" + fx + ", " + fy + ", " + fz + ")");
	}

	public void printPosition() {
		commandQueue.readBuffer(bodiesXBuffer);
		commandQueue.readBuffer(bodiesYBuffer);
		commandQueue.readBuffer(bodiesZBuffer);
		System.out.println("bodiesX: " + Arrays.toString(bodiesXBuffer.getData()));
		System.out.println("bodiesY: " + Arrays.toString(bodiesYBuffer.getData()));
		System.out.println("bodiesZ: " + Arrays.toString(bodiesZBuffer.getData()));
	}

	@Override
	public int getNumberOfBodies() {
		return nbodies;
	}

	/**
	 * Initializes the given context properties so that they may be used to create an OpenCL context for the given GL object.
	 * 
	 * @param contextProperties
	 *            The context properties
	 * @param gl
	 *            The GL object
	 */
	private static void initContextProperties(final cl_context_properties contextProperties, final GL gl) {
		// Adapted from http://jogamp.org/jocl/www/

		final GLContext glContext = gl.getContext();
		if (!glContext.isCurrent()) {
			throw new IllegalArgumentException("OpenGL context is not current. This method should be called " + "from the OpenGL rendering thread, when the context is current.");
		}

		final long glContextHandle = glContext.getHandle();
		final GLContextImpl glContextImpl = (GLContextImpl) glContext;
		final GLDrawableImpl glDrawableImpl = glContextImpl.getDrawableImpl();
		final NativeSurface nativeSurface = glDrawableImpl.getNativeSurface();

		if (glContext instanceof X11GLXContext) {
			final long displayHandle = nativeSurface.getDisplayHandle();
			contextProperties.addProperty(CL_GL_CONTEXT_KHR, glContextHandle);
			contextProperties.addProperty(CL_GLX_DISPLAY_KHR, displayHandle);
		} else if (glContext instanceof WindowsWGLContext) {
			final long surfaceHandle = nativeSurface.getSurfaceHandle();
			contextProperties.addProperty(CL_GL_CONTEXT_KHR, glContextHandle);
			contextProperties.addProperty(CL_WGL_HDC_KHR, surfaceHandle);
		} else if (glContext instanceof MacOSXCGLContext) {
			contextProperties.addProperty(CL_CGL_SHAREGROUP_KHR, glContextHandle);
		} else if (glContext instanceof EGLContext) {
			final long displayHandle = nativeSurface.getDisplayHandle();
			contextProperties.addProperty(CL_GL_CONTEXT_KHR, glContextHandle);
			contextProperties.addProperty(CL_EGL_DISPLAY_KHR, displayHandle);
		} else {
			throw new RuntimeException("unsupported GLContext: " + glContext);
		}
	}

	public static void main(final String[] args) {
		final int nbodies = 128;
		final GpuNBodySimulation nBodySimulation = new GpuNBodySimulation(Mode.DEFAULT, nbodies, new PlummerUniverseGenerator());

		nBodySimulation.init(null);
		nBodySimulation.initPositionBuffer(null, -1);

		if (HOST_DEBUG) {
			nBodySimulation.printPosition();
		}
		for (int i = 0; i < 5000; ++i) {
			nBodySimulation.step();

			if (HOST_DEBUG) {
//				nBodySimulation.printChildren();
//				nBodySimulation.printPosition();
				nBodySimulation.printTotalForce();
				nBodySimulation.printEnergy();
			}

			Thread.yield();
		}

		if (HOST_DEBUG) {
			nBodySimulation.printPosition();
			nBodySimulation.printTotalForce();
			System.out.println("done");
		}

	}

}
