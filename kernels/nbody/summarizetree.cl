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
	
	DEBUG_PRINT(("- Info Summarizetree  -\n"));
	int bottom = *_bottom;
	DEBUG_PRINT(("bottom: %d\n", bottom));
	int stepSize = get_local_size(0) * get_num_groups(0);

    // align to warp size
    int node = (bottom & (-WARPSIZE)) + get_global_id(0);
    if (node < bottom)
        node += stepSize;
	
	DEBUG_PRINT(("start node: %d\n", node));
	
	int missing = 0;
	
	int cellBodyCount = 0;
	float cellMass;
	float centerX, centerY, centerZ;
	
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
		    	DEBUG_PRINT(("\tchildIndex: %d\n", childIndex));
		    	DEBUG_PRINT(("\t\tchildIndex: %d\n", child));
				
				// is used
		    	if (child >= 0) {
					DEBUG_PRINT(("\t\tchild is used\n"));
		    		if (childIndex != usedChildIndex) {
						DEBUG_PRINT(("\t\tmove to front\n"));
		    			_child[NUMBER_OF_CELLS * node + childIndex] = -1;
		    			_child[NUMBER_OF_CELLS * node + usedChildIndex] = child;
		    		}

					float mass = _mass[child];
					DEBUG_PRINT(("\t\tmass[child]: %f\n", mass));
					DEBUG_PRINT(("\t\t\tchild: %d\n", child));
				
					// Cache missing children
					_child[WORKGROUP_SIZE * missing + get_local_id(0)] = child;
					
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
				int child = _child[(missing - 1) * WORKGROUP_SIZE + get_local_id(0)];
				DEBUG_PRINT(("\t\tchild: %d\n", child));
				float mass = _mass[child];
				
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
			} while ((cellMass >= 0.0f) && (missing != 0));
		}
	
		if (missing == 0) {
			DEBUG_PRINT(("\tmissing is zero\n"));
			DEBUG_PRINT(("\t\tbodyCount: %d\n", cellBodyCount));
			DEBUG_PRINT(("\t\tcellMass: %f\n", cellMass));
			_bodyCount[node] = cellBodyCount;
			float inverseMass = 1.0f / cellMass;
			_posX[node] = centerX * inverseMass;
			_posY[node] = centerY * inverseMass;
			_posZ[node] = centerZ * inverseMass;
			
			// make sure data is visible before setting mass
			atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_seq_cst, memory_scope_device);
		
			_mass[node] = cellMass;
			
			node += stepSize; // next cell
		}
	}
}