package ch.fhnw.woipv.nbody.internal.opencl;

import static org.jocl.CL.*;

import org.jocl.cl_command_queue;

public class CLCommandQueue {
	private cl_command_queue queue;
	
	public CLCommandQueue(cl_command_queue queue) {
		this.queue = queue;
	}
	
	public void execute(CLKernel kernel, int dimensions, long globalWorkSize, long localWorkSize) {
		clEnqueueNDRangeKernel(queue, kernel.getKernel(), dimensions, null, new long[] { globalWorkSize }, new long[] { localWorkSize }, 0, null, null);
	}
	
	public void readBuffer(CLMemory memory) {
		clEnqueueReadBuffer(queue, memory.getMemory(), CL_TRUE, 0, memory.getSize(), memory.getPointer(), 0, null, null);
	}
	
	public void finish() {
		clFinish(queue);
	}
}
