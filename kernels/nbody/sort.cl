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
	__global int* _blockCount, __global int* _bodyCount,  __global float* _radius, __global int* _bottom, __global float* _mass, __global int* _child) {

    int bottom = *_bottom;
    int stepSize = get_local_size(0) * get_num_groups(0);
    int cell = NUMBER_OF_NODES + 1 - stepSize + get_global_id(0);

	// Iterate over all cells assigned to thread
    while (cell >= bottom) {
        int start = _start[cell];
        if (start >= 0) {
            #pragma unroll NUMBER_OF_CELLS
            for (int i = 0; i < NUMBER_OF_CELLS; ++i)
            {
                int ch = _child[NUMBER_OF_CELLS * cell + i];
                if (ch >= NBODY) { // child is a cell
                    _start[ch] = start;  /* Set start ID of child */
                    start += _count[ch]; /* Add #bodies in subtree */
                } else if (ch >= 0) { // child is a body
                    _sort[start] = ch;   /* Record body in sorted array */
                    ++start;
                }
            }
        }

        cell -= stepSize;  /* Move on to next cell */
    }
}