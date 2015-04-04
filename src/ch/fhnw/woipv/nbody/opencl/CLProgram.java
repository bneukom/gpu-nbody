package ch.fhnw.woipv.nbody.opencl;

import java.io.Closeable;
import java.io.IOException;

import org.jocl.cl_program;

import static org.jocl.CL.*;

public class CLProgram implements Closeable {
	private final cl_program program;
	
	public CLProgram(final cl_program program) {
		this.program = program;
	}

	public void build(final String args) {
		clBuildProgram(program, 0, null, "-cl-std=CL2.0", null, null);
	}
	
	public CLKernel createKernel(String kernelName) {
		return CLKernel.createKernel(this, kernelName);
	}
	
	public cl_program getId() {
		return program;
	}

	@Override
	public void close() throws IOException {
//		clReleaseProgram(program);
	}

	
	
}
