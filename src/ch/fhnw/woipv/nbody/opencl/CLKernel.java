package ch.fhnw.woipv.nbody.opencl;

import static org.jocl.CL.*;

import java.io.Closeable;
import java.io.IOException;

import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_kernel;

public class CLKernel implements Closeable {
	private cl_kernel kernel;
	private int argumentIndex;

	public CLKernel(cl_kernel kernel) {
		this.kernel = kernel;
	}

	public static CLKernel createKernel(CLProgram program, String kernelName) {
		cl_kernel kernel = clCreateKernel(program.getId(), kernelName, null);

		return new CLKernel(kernel);
	}

	public void setArgument(int index, long argSize, Pointer value) {
		throw new IllegalStateException();
	}

	public cl_kernel getKernel() {
		return kernel;
	}

	public CLKernel addArgument(CLMemory memory) {
		clSetKernelArg(kernel, argumentIndex++, Sizeof.cl_mem, Pointer.to(memory.getMemory()));
		return this;
	}

	@Override
	public void close() throws IOException {
//		clReleaseKernel(kernel);
	}
}
