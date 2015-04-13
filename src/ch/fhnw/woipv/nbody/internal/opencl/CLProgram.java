package ch.fhnw.woipv.nbody.internal.opencl;

import java.io.Closeable;
import java.io.IOException;
import java.util.Arrays;

import org.jocl.cl_program;

import static org.jocl.CL.*;

public class CLProgram implements Closeable {
	private final cl_program program;
	
	public CLProgram(final cl_program program) {
		this.program = program;
	}

	public void build(final BuildOption... options) {
		clBuildProgram(program, 0, null, Arrays.stream(options).map(o -> o.option).reduce("", (accu, o) -> accu + " " + o), null, null);
	}

	public CLKernel createKernel(String kernelName) {
		return CLKernel.createKernel(this, kernelName);
	}

	public cl_program getId() {
		return program;
	}

	@Override
	public void close() throws IOException {
		clReleaseProgram(program);
	}

	public static class BuildOption {
		private final String option;
		public static final BuildOption CL20 = new BuildOption("-cl-std=CL2.0");
		public static final BuildOption MAD = new BuildOption("-cl-mad-enable");
		
		public BuildOption(String option) {
			this.option = option;
		}
	}
	
}
