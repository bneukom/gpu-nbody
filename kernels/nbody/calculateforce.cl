#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

#include "kernels/nbody/debug.h"

#define NULL_BODY (-1)
#define LOCK (-2)

// TODO pass as argument
#define WARPSIZE 64
#define MAXDEPTH 64

#define NUMBER_OF_CELLS 8 // the number of cells per node

__attribute__ ((reqd_work_group_size(WORKGROUP_SIZE, 1, 1)))
__kernel void summarizeTree(
	__global float* _posX, __global float* _posY, __global float* _posZ, 
	__global float* _velX, __global float* _velY, __global float* _velZ, 
	__global float* _accX, __global float* _accY, __global float* _accZ, 
	__global int* _blockCount, __global int* _bodyCount,  __global float* _radius, __global int* _maxDepth, __global float* epsilonSquared, __global float *
	__global int* _bottom, __global float* _mass, __global int* _child, __global int* _start, __global int* _sorted) {
	
	local volatile int localPos[MAX_DEPTH * WORKGROUP_SIZE / WARPSIZE];
	local volatile int localNode[MAX_DEPTH * WORKGROUP_SIZE / WARPSIZE];
	
	local volatile float dq[MAX_DEPTH * WORKGROUP_SIZE / WARPSIZE];
	
	if (get_local_id(0) == 0) {
		float radiusTheta = *_radius / THETA;
		dq[0] = radiusTheta * radiusTheta; 
		for (int i = 1; i < maxDepth; ++i) {
			dq[i] = 0.25f * dq[i - 1];		
		}
		
		if (maxDepth > MAXDEPTH) {
			printf("ERROR: maxDepth");
		}
	}
	
	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
	
	if (maxDepth <= MAX_DEPTH) {
		int base = get_local_id(0) / WARPSIZE;
		int sBase = base * WARPSIZE;
		int j = base * MAXDEPTH;
		
		// index in warp
		int diff = get_local_id(0) - sbase;
		
		// make multiple copies to avoid index calculations later
	    if (diff < MAXDEPTH) {
			dq[diff+j] = dq[diff];
	    }
		
		barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
		
		for (int bodyIndex = get_global_id(0); bodyIndex < NBODIES; bodyIndex += get_local_size(0) * get_num_groups(0)) {
			int sortedIndex = _sorted[bodyIndex];
			
			float posX = _posX[sortedIndex];
			float posY = _posY[sortedIndex];
			float posZ = _posZ[sortedIndex];
			
			float accX = 0.0f;
			float accY = 0.0f;
			float accZ = 0.0f;
			
			// initialize stack
			int depth = j;
			if (sBase == get_local_id(0)) {
				node[depth] = NUMBER_OF_NODES;
				pos[depth] = 0;
			}
			
			mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
			
			while (depth >= j) {
			
			
			}
			
		}
	}
}