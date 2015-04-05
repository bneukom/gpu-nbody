package ch.fhnw.woipv.nbody.internal.opencl;

import static org.jocl.CL.*;

import java.util.Arrays;
import java.util.List;
import java.util.function.Predicate;
import java.util.stream.Collectors;

import org.jocl.Pointer;
import org.jocl.cl_context;
import org.jocl.cl_context_properties;
import org.jocl.cl_device_id;

public class CLDevice {
	private cl_device_id id;
	private CLPlatform platform;
	
	public CLDevice(cl_device_id id, CLPlatform platform) {
		this.id = id;
		this.platform = platform;
	}
	
	public String getDeviceInfo(int param) {
		long size[] = new long[1];
		clGetDeviceInfo(id, param, 0, null, size);

		byte buffer[] = new byte[(int) size[0]];
		clGetDeviceInfo(id, param, buffer.length, Pointer.to(buffer), null);

		return new String(buffer, 0, buffer.length - 1);
	}
	
	public CLContext createContext() {
		final cl_context_properties contextProperties = new cl_context_properties();
		contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform.getPlatformId());
		
		final cl_context context = clCreateContext(contextProperties, 1, new cl_device_id[] { id }, null, null, null);

		return new CLContext(context, this);
	}

	public cl_device_id getId() {
		return id;
	}

}
