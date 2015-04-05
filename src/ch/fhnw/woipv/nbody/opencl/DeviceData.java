package ch.fhnw.woipv.nbody.opencl;

import ch.fhnw.woipv.nbody.internal.opencl.CLCommandQueue;
import ch.fhnw.woipv.nbody.internal.opencl.CLMemory;

public interface DeviceData {
	public CLMemory allocateMemory();
	public void read(CLCommandQueue queue);
}
