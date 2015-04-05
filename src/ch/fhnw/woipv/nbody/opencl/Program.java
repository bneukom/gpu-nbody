package ch.fhnw.woipv.nbody.opencl;

import java.util.List;
import java.util.Map;

import ch.fhnw.woipv.nbody.internal.opencl.CLKernel;
import ch.fhnw.woipv.nbody.internal.opencl.CLProgram;

public class Program {
	private Map<CLKernel, Integer> localDimension;
	private Map<CLKernel, Integer> globalDimension;
	private List<CLKernel> kernels;
	private boolean debug;
	
	private CLProgram program;
	
	public void attachKernel(String kernel, int localWorkSize, int globalWorkSize) {
		
	}
	
	public void loadProgram(boolean debug) {
		
	}
	
	@FunctionalInterface
	private static interface ArgumentSupplier {
		public void attach(CLKernel kernel);
	}
	
	private static class Kernel {
		private int localDimension;
		private int globalDimension;
		private ArgumentSupplier supplier;
	}
}
