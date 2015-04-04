package ch.fhnw.woipv.nbody.opencl;

import static org.jocl.CL.*;

import org.jocl.cl_command_queue;

public class CLCommandQueue {
	private cl_command_queue queue;
	
	public CLCommandQueue(cl_command_queue queue) {
		this.queue = queue;
	}
	
}
