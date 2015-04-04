package ch.fhnw.woipv.nbody.opencl;

import java.io.Closeable;
import java.io.IOException;

import static org.jocl.CL.*;

import org.jocl.Pointer;
import org.jocl.cl_mem;

public class CLMemory implements Closeable {
	private cl_mem memory;
	private Pointer pointer;
	private long size;
	
	public CLMemory(cl_mem memory, long size, Pointer pointer) {
		super();
		this.memory = memory;
		this.size = size;
		this.pointer = pointer;
	}

	public Pointer getPointer() {
		return pointer;
	}
	
	public long getSize() {
		return size;
	}
	
	public cl_mem getMemory() {
		return memory;
	}

	@Override
	public void close() throws IOException {
		clReleaseMemObject(memory);
	}
}
