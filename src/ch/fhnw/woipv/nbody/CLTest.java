package ch.fhnw.woipv.nbody;

import static org.jocl.CL.*;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;

import ch.fhnw.woipv.nbody.opencl.CLCommandQueue;
import ch.fhnw.woipv.nbody.opencl.CLContext;
import ch.fhnw.woipv.nbody.opencl.CLDevice;
import ch.fhnw.woipv.nbody.opencl.CLKernel;
import ch.fhnw.woipv.nbody.opencl.CLMemory;
import ch.fhnw.woipv.nbody.opencl.CLPlatform;
import ch.fhnw.woipv.nbody.opencl.CLProgram;

public class CLTest {
	public static void main(final String[] args) throws IOException {
		final int numWorkGroups = 10;
		final int localWorkSize = 16;
		final int globalWorkSize = numWorkGroups * localWorkSize;

		CL.setExceptionsEnabled(true);

		final CLPlatform platform = CLPlatform.getPlatforms().get(0);

		final CLDevice device = platform.getDevice(CL_DEVICE_TYPE_ALL, d -> {
			final String deviceVersion = d.getDeviceInfo(CL_DEVICE_VERSION);
			final String versionString = deviceVersion.substring(7, 10);
			final float version = Float.parseFloat(versionString);
			final String deviceName = d.getDeviceInfo(CL_DEVICE_NAME);
			return version >= 2.0 && !deviceName.contains("CPU");
		});

		final CLContext context = device.createContext();
		final CLCommandQueue commandQueue = context.createCommandQueue();

		final int tmpArray[] = new int[globalWorkSize];
		final CLProgram program = context.createProgram(Files.readAllLines(new File("kernels/test.cl").toPath()).stream()
				.reduce("", (accu, l) -> accu + l + System.lineSeparator()));

		program.build("-cl-std=CL2.0");

		final CLKernel kernel = program.createKernel("test");
		final CLMemory clMemory = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, tmpArray);

		kernel.addArgument(clMemory);

		commandQueue.execute(kernel, 1, globalWorkSize, localWorkSize);

		commandQueue.read(clMemory);

		commandQueue.finish();

		System.out.println(tmpArray[3]);
	}

}
