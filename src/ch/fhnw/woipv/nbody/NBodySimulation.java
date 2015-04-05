package ch.fhnw.woipv.nbody;

import java.util.ArrayList;
import java.util.List;

import ch.fhnw.woipv.nbody.internal.opencl.CLCommandQueue;
import ch.fhnw.woipv.nbody.internal.opencl.CLMemory;
import ch.fhnw.woipv.nbody.opencl.DeviceData;

public class NBodySimulation implements DeviceData {
	private float[] min = new float[3];
	private List<Body> bodies = new ArrayList<Body>();
	
	@Override
	public CLMemory allocateMemory() {
		return null;
	}

	@Override
	public void read(CLCommandQueue queue) {

	}
}
