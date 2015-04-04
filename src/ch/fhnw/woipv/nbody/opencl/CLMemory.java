package ch.fhnw.woipv.nbody.opencl;

import java.io.Closeable;
import java.io.IOException;

import static org.jocl.CL.*;

import org.jocl.Pointer;
import org.jocl.cl_mem;

public class CLMemory implements Closeable {
	private cl_mem memory;
	
	public CLMemory(cl_mem memory) {
		super();
		this.memory = memory;
	}

	public Pointer createPointer() {
		return Pointer.to(memory);
	}
	
	public cl_mem getMemory() {
		return memory;
	}

	@Override
	public void close() throws IOException {
		clReleaseMemObject(memory);
	}
}
