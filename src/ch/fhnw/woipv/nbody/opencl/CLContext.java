package ch.fhnw.woipv.nbody.opencl;

import static org.jocl.CL.*;

import java.io.Closeable;
import java.io.IOException;

import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_mem;
import org.jocl.cl_program;

public class CLContext implements Closeable {
	private final cl_context context;
	private final CLDevice device;
	
	public CLContext(final cl_context context, final CLDevice device) {
		this.context = context;
		this.device = device;
	}
	
	public CLCommandQueue createCommandQueue() {
		@SuppressWarnings("deprecation")
		final cl_command_queue commandQueue = clCreateCommandQueue(context, device.getId(), 0, null);
		return new CLCommandQueue(commandQueue);
	}
	
	public CLProgram createProgram(final String source) {
		final cl_program program = clCreateProgramWithSource(context, 1, new String[] { source }, null, null);
		
		return new CLProgram(program);
	}
	
	public CLMemory createBuffer(final long flags, final int[] data) {
		final Pointer pointer = Pointer.to(data);
		final cl_mem mem = clCreateBuffer(context, flags, Sizeof.cl_uint * data.length, pointer, null);
		return new CLMemory(mem,  Sizeof.cl_uint * data.length, pointer);
	}

	public cl_context getContext() {
		return context;
		
	}

	@Override
	public void close() throws IOException {
 		clReleaseContext(context);
	}
}
