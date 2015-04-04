package ch.fhnw.woipv.nbody.opencl;

import static org.jocl.CL.*;

import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_kernel;

public class CLKernel {
	private cl_kernel kernel;
	
	public CLKernel(cl_kernel kernel) {
		this.kernel = kernel;
	}

	public static CLKernel createKernel(CLProgram program, String kernelName) {
		cl_kernel kernel = clCreateKernel(program.getId(), kernelName, null);
	
		
		return new CLKernel(kernel);
	}
	
	public void setArgument(int index, long argSize, Pointer value) {
		
	}
	
	public CLKernel addArgument(CLMemory memory) {
		clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(memory.getMemory()));
		return this;
	}
}
