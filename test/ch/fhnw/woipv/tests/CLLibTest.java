package ch.fhnw.woipv.tests;

import static org.jocl.CL.*;

import java.io.File;
import java.io.IOException;

import net.benjaminneukom.oocl.cl.OOCL;
import net.benjaminneukom.oocl.cl.CLCommandQueue;
import net.benjaminneukom.oocl.cl.CLContext;
import net.benjaminneukom.oocl.cl.CLDevice;
import net.benjaminneukom.oocl.cl.CLKernel;
import net.benjaminneukom.oocl.cl.CLMemory;
import net.benjaminneukom.oocl.cl.CLProgram;
import net.benjaminneukom.oocl.cl.CLProgram.BuildOption;

public class CLLibTest {
	public static void main(final String[] args) throws IOException {
		final int numWorkGroups = 10;
		final int localWorkSize = 10;
		final int globalWorkSize = numWorkGroups * localWorkSize;

		final CLDevice device = OOCL.createDevice();

		final CLContext context = device.createContext();
		final CLCommandQueue commandQueue = context.createCommandQueue();

		final CLProgram program = context.createProgram(new File("kernels/test/libTest.cl"));

		program.build(BuildOption.CL20, BuildOption.MAD);

		final CLKernel kernel = program.createKernel("test");

		final float tmpArray[] = new float[globalWorkSize];
		final CLMemory clMemory = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, tmpArray);

		kernel.addArgument(clMemory);

		commandQueue.execute(kernel, 1, globalWorkSize, localWorkSize);
		commandQueue.finish();
		
//		commandQueue.execute(kernel, 1, globalWorkSize, localWorkSize);
		commandQueue.readBuffer(clMemory);
		System.out.println(tmpArray[3]);
//		commandQueue.finish();

	}

}
