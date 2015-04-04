package ch.fhnw.woipv.nbody;

import static org.jocl.CL.*;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;

import org.jocl.Pointer;
import org.jocl.Sizeof;

import ch.fhnw.woipv.nbody.opencl.CLCommandQueue;
import ch.fhnw.woipv.nbody.opencl.CLContext;
import ch.fhnw.woipv.nbody.opencl.CLDevice;
import ch.fhnw.woipv.nbody.opencl.CLKernel;
import ch.fhnw.woipv.nbody.opencl.CLMemory;
import ch.fhnw.woipv.nbody.opencl.CLPlatform;
import ch.fhnw.woipv.nbody.opencl.CLProgram;

public class NBody {
	public static void main(final String[] args) throws IOException {
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
		
		final CLProgram program = context.createProgram(Files.readAllLines(new File("kernels/fenceTest.cl").toPath()).stream().reduce("", (accu, l) -> accu + l + System.lineSeparator()));
	
		program.build("-cl-std=CL2.0");
		
		final CLKernel kernel = program.createKernel("sampleKernel");
		
		final int tmpArray[] = new int[1];
		final Pointer tmpPointer = Pointer.to(tmpArray);
		
		// mainasd
		try (CLMemory tmp = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, Sizeof.cl_uint, tmpPointer)){
			kernel.addArgument(tmp);
		}
	}

}
