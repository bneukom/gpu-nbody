#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

#include "kernels/nbody/debug.h"

#define NULL_BODY (-1)
#define LOCK (-2)

// TODO pass as argument
#define WARPSIZE 64

#define NUMBER_OF_CELLS 8 // the number of cells per node

__attribute__ ((reqd_work_group_size(WORKGROUP_SIZE, 1, 1)))
__kernel void summarizeTree(
	__global float* _posX, __global float* _posY, __global float* _posZ, 
	__global int* _blockCount, __global int* _bodyCount,  __global float* _radius, __global int* _bottom, __global float* _mass, __global int* _child) {
	
	// TODO
	__local volatile int localChild[WORKGROUP_SIZE * NUMBER_OF_CELLS];
	
	DEBUG_PRINT(("- Info Summarizetree  -\n"));
	int bottom = *_bottom;
	DEBUG_PRINT(("bottom: %d\n", bottom));
	int stepSize = get_local_size(0) * get_num_groups(0);
	DEBUG_PRINT(("stepSize: %d\n", stepSize));
	DEBUG_PRINT(("NUMBER_OF_NODES: %d\n", NUMBER_OF_NODES));

	// TODO small work group sizes DO NOT WORK WITH this?!
    // align to warp size
    int node = (bottom & (-WARPSIZE)) + get_global_id(0);
    if (node < bottom)
        node += stepSize;
	
	DEBUG_PRINT(("start node: %d\n", node));
	
	int missing = 0;
	
	int cellBodyCount = 0;
	float cellMass;
	float mass;
	float centerX, centerY, centerZ;
	
	DEBUG_PRINT(("- Code Summarizetree  -\n"));
	// iterate over all cells assigned to thread
	while (node <= NUMBER_OF_NODES) {
		if (missing == 0) {
			DEBUG_PRINT(("\tnew cell - initialize\n"));
			// new cell, so initialize
			cellMass = 0.0f;
			centerX = 0.0f;
		    centerY = 0.0f;
		    centerZ = 0.0f;
		    cellBodyCount = 0;
		    
		    // gets incremented when cell is used
		    int usedChildIndex = 0; 
		    
		    #pragma unroll NUMBER_OF_CELLS
		    for (int childIndex = 0; childIndex < NUMBER_OF_CELLS; childIndex++) {
		    	int child = _child[node * NUMBER_OF_CELLS + childIndex];
		    	DEBUG_PRINT(("\t\tchildIndex: %d\n", childIndex));
		    	DEBUG_PRINT(("\t\t\tchild: %d\n", child));
				
				// is used
		    	if (child >= 0) {
					DEBUG_PRINT(("\t\t\tchild is used\n"));
		    		if (childIndex != usedChildIndex) {
						DEBUG_PRINT(("\t\t\tmove to front\n"));
		    			_child[NUMBER_OF_CELLS * node + childIndex] = -1;
		    			_child[NUMBER_OF_CELLS * node + usedChildIndex] = child;
		    		}
		    		
					// Cache missing children
					localChild[WORKGROUP_SIZE * missing + get_local_id(0)] = child;
					
					mass = _mass[child];
					DEBUG_PRINT(("\t\t\tmass[child]: %f\n", mass));
					DEBUG_PRINT(("\t\t\t\tchild: %d\n", child));

					++missing;
					
					if (mass >= 0.0f) {
						// child is ready	
						--missing;
						
						if (child >= NBODIES) {
							cellBodyCount += _bodyCount[child] - 1;
						}
						
						cellMass += mass;
						centerX += _posX[child] * mass;
						centerY += _posY[child] * mass;
						centerZ += _posZ[child] * mass;
					}
					usedChildIndex++;
		    	}		    	
		    }
		
			cellBodyCount += usedChildIndex;
	
		}
	
		if (missing != 0) {
			DEBUG_PRINT(("\tmissing is not zero - work missing children\n"));
			do {
				int child = localChild[(missing - 1) * WORKGROUP_SIZE + get_local_id(0)];
				DEBUG_PRINT(("\t\tchild: %d\n", child));
				mass = _mass[child];
				
				// Body children can never be missing, so this is a cell
				if (mass >= 0.0f) {
					--missing;
					
					if (child >= NBODIES) {
						cellBodyCount += _bodyCount[child] - 1;
					}
					
					cellMass += mass;
					centerX += _posX[child] * mass;
					centerY += _posY[child] * mass;
					centerZ += _posZ[child] * mass;

				}
			} while ((mass >= 0.0f) && (missing != 0));
		}
	
		if (missing == 0) {
			DEBUG_PRINT(("\tmissing is zero\n"));
			DEBUG_PRINT(("\t\tbodyCount: %d\n", cellBodyCount));
			DEBUG_PRINT(("\t\tcellMass: %f\n", cellMass));
			_bodyCount[node] = cellBodyCount;
			mass = 1.0f / cellMass;
			DEBUG_PRINT(("\t\tcenterX: %f\n", centerX));
			DEBUG_PRINT(("\t\tcenterY: %f\n", centerY));
			DEBUG_PRINT(("\t\tcenterZ: %f\n", centerZ));
			_posX[node] = centerX * mass;
			_posY[node] = centerY * mass;
			_posZ[node] = centerZ * mass;
			
			// make sure data is visible before setting mass
			atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_seq_cst, memory_scope_device);
		
			_mass[node] = cellMass;
			
			node += stepSize; // next cell
			DEBUG_PRINT(("\t\tnext node: %d\n", node));
		}
	}
}