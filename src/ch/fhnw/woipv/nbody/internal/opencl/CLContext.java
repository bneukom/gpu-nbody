package ch.fhnw.woipv.nbody.internal.opencl;

import static org.jocl.CL.*;

import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.stream.Collectors;

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

	public CLProgram createProgram(final File... file) throws IOException {
		final String[] programms = Arrays.stream(file).map(f -> {
			try {
				return Files.readAllLines(f.toPath()).stream().reduce("", (accu, l) -> accu + l + System.lineSeparator());
			} catch (Exception e) {
				e.printStackTrace();
			}
			return null;
		}).toArray(String[]::new);

		return createProgram(programms);
	}

	public CLProgram createProgram(final String... sources) {

		final cl_program program = clCreateProgramWithSource(context, 1, sources, null, null);

		return new CLProgram(program);
	}

	public CLMemory createBuffer(final long flags, final int[] data) {
		final Pointer pointer = Pointer.to(data);
		final cl_mem mem = clCreateBuffer(context, flags, Sizeof.cl_int * data.length, pointer, null);
		return new CLMemory(mem, Sizeof.cl_int * data.length, pointer);
	}
	
	public CLMemory createBuffer(final long flags, final long[] data) {
		final Pointer pointer = Pointer.to(data);
		final cl_mem mem = clCreateBuffer(context, flags, Sizeof.cl_ulong * data.length, pointer, null);
		return new CLMemory(mem, Sizeof.cl_ulong * data.length, pointer);
	}

	public CLMemory createBuffer(long flags, float[] data) {
		final Pointer pointer = Pointer.to(data);
		final cl_mem mem = clCreateBuffer(context, flags, Sizeof.cl_float * data.length, pointer, null);
		return new CLMemory(mem, Sizeof.cl_float * data.length, pointer);

	}

	public cl_context getContext() {
		return context;

	}

	@Override
	public void close() throws IOException {
		clReleaseContext(context);
	}

}
