package ch.fhnw.woipv.nbody.boundsReduction;

import ch.fhnw.woipv.nbody.NBodySimulation;
import ch.fhnw.woipv.nbody.opencl.Program;

public class BoundsReductionTest {
	public static void main(String[] args) {
		Program p = new Program("kernels/nbody/boundingbox.cl", true);
		NBodySimulation simulation = new NBodySimulation();
		
		p.attachKernel("boundingBox", 10, 10, s -> {
			
		});
		
		p.execute();
		p.read();
	}
}
