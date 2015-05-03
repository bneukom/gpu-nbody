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
__kernel void sort(
	__global float* _posX, __global float* _posY, __global float* _posZ,
	__global float* _velX, __global float* _velY, __global float* _velZ, 
	__global float* _accX, __global float* _accY, __global float* _accZ,  
	__global int* _step, __global int* _blockCount, __global int* _bodyCount,  __global float* _radius, __global int* _maxDepth,
	__global int* _bottom, __global float* _mass, __global int* _child, __global volatile atomic_int* _start, __global int* _sorted) {
	
	DEBUG_PRINT((" - Info Sort -\n"));
	DEBUG_PRINT(("workDim: %d\n", get_work_dim())); 
	DEBUG_PRINT(("globalId: %d\n", get_global_id(0))); 
	DEBUG_PRINT(("localId: %d\n", get_local_id(0))); 
    int bottom = *_bottom;
    int stepSize = get_local_size(0) * get_num_groups(0);
    int cell = NUMBER_OF_NODES + 1 - stepSize + get_global_id(0);
	DEBUG_PRINT(("bottom: %d (%d, %d)\n", bottom, get_local_id(0), get_global_id(0)));
	DEBUG_PRINT(("stepSize: %d (%d, %d)\n", stepSize,  get_local_id(0), get_global_id(0)));
	DEBUG_PRINT(("start  cell: %d (%d, %d)\n", cell,  get_local_id(0), get_global_id(0)));
	// Iterate over all cells assigned to thread
    while (cell >= bottom) {
    	DEBUG_PRINT(("cell1: %d (%d, %d)\n", cell, get_local_id(0), get_global_id(0)));
		
		// TODO do we need atomics?
        //int start = _start[cell];
        int start = atomic_load_explicit(&_start[cell], memory_order_acquire, memory_scope_device);
        
        DEBUG_PRINT(("start: %d (%d, %d)\n", start, get_local_id(0), get_global_id(0)));
        if (start >= 0) {
            #pragma unroll NUMBER_OF_CELLS
            for (int i = 0; i < NUMBER_OF_CELLS; ++i) {
                int ch = _child[NUMBER_OF_CELLS * cell + i];
                DEBUG_PRINT(("\tch: %d (%d, %d)\n", ch, get_local_id(0), get_global_id(0)));
                if (ch >= NBODIES) { // child is a cell
               	 	DEBUG_PRINT(("\t\tchild is cell\n"));
               	 	
               	 	// TODO do we need atomics?
                	// set start id of child
                    // _start[ch] = start; 
                    atomic_store_explicit (&_start[ch], start, memory_order_release, memory_scope_device);	
                    
                    // TODO don't we need an atomic_work_item_fence for visibility here?
                    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE, memory_order_seq_cst, memory_scope_device);
		
                    // Add #bodies in subtree
                    start += _bodyCount[ch];
                    DEBUG_PRINT(("\t\t\tstart:%d\n", start));
                    DEBUG_PRINT(("\t\t\t_bodyCount[ch]:%d\n", _bodyCount[ch]));
                } else if (ch >= 0) { // child is a body
                 	DEBUG_PRINT(("\t\tchild is body\n"));
                	// Record body in sorted array
                    _sorted[start] = ch;
                    DEBUG_PRINT(("\t\t\tstart:%d\n", start));
                    
                    ++start;
                }
            }
            
            DEBUG_PRINT(("cell(%d) - stepSize(%d) (%d, %d)\n", cell, stepSize, get_local_id(0), get_global_id(0)));
            cell -= stepSize;  /* Move on to next cell */
            DEBUG_PRINT(("cell2: %d (%d, %d)\n", cell, get_local_id(0), get_global_id(0)));
        }

       
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); // throttle
    }
}