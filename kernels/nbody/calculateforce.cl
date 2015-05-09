#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

#include "kernels/nbody/debug.h"

#define NULL_BODY (-1)
#define LOCK (-2)

// TODO pass as argument
#define WARPSIZE 64
#define MAXDEPTH 64
#define EPSILON (0.05f * 0.05f)
#define THETA (0.5f * 0.5f)
//#define TIMESTEP 0.025f
#define TIMESTEP 0.00078125f

#define NUMBER_OF_CELLS 8 // the number of cells per node

__attribute__ ((reqd_work_group_size(WORKGROUP_SIZE, 1, 1)))
__kernel void calculateForce(
	__global float* _posX, __global float* _posY, __global float* _posZ, 
	__global float* _velX, __global float* _velY, __global float* _velZ, 
	__global float* _accX, __global float* _accY, __global float* _accZ, 
	__global int* _step, __global int* _blockCount, __global int* _bodyCount,  __global float* _radius, __global int* _maxDepth, 
	__global int* _bottom, __global float* _mass, __global int* _child, __global int* _start, __global int* _sorted, __global int* _error) {
	
	local volatile int localPos[MAXDEPTH * WORKGROUP_SIZE / WARPSIZE];
	local volatile int localNode[MAXDEPTH * WORKGROUP_SIZE / WARPSIZE];
	
	local volatile float dq[MAXDEPTH * WORKGROUP_SIZE / WARPSIZE];
	
	// TODO itolsqd is 1/THETA ?
	DEBUG_PRINT(("- Info Calculate Force  -\n"));
	DEBUG_PRINT(("stack size: %d\n", MAXDEPTH * WORKGROUP_SIZE / WARPSIZE));
	DEBUG_PRINT(("epsilon %f\n", EPSILON));
	DEBUG_PRINT(("theta %f\n", THETA));
	DEBUG_PRINT(("MAXDEPTH %d\n", MAXDEPTH));
	DEBUG_PRINT(("maxDepth %d\n", *_maxDepth));
	
	if (get_local_id(0) == 0) {
		float radius = *_radius; 
		dq[0] = radius * radius / THETA;
		
		int i;
		for (i = 1; i < *_maxDepth; ++i) {
			dq[i] = 0.25f * dq[i - 1];	
			dq[i - 1] += EPSILON;	
		}
		dq[i - 1] += EPSILON;
		
		if (*_maxDepth > MAXDEPTH) {
			printf("ERROR: maxDepth\n");
			*_error = 1;
			return;
		} 
	}
	
	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
	
	if (*_maxDepth <= MAXDEPTH) {
		int base = get_local_id(0) / WARPSIZE;
		int sBase = base * WARPSIZE;
		int j = base * MAXDEPTH;
		
		// index in warp
		int diff = get_local_id(0) - sBase;
		
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
				localNode[depth] = NUMBER_OF_NODES;
				localPos[depth] = 0;
			}
			
			mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
			
			while (depth >= j) {
				int top;
				while ((top = localPos[depth]) < 8) {
					int child = _child[localNode[depth] * NUMBER_OF_CELLS + top];
					
					if (sBase == get_local_id(0)) {
						// first thread in warp
						localPos[depth] = top + 1;
					}
					
					// TODO memfence needed?
					mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
					
					if (child >= 0) {
						float distX = _posX[child] - posX;
						float distY = _posX[child] - posY;
						float distZ = _posX[child] - posZ;
						
						// squared distance plus softening
						float distSquared = distX * distX + distY * distY + distZ * distZ + EPSILON;
						
						if ((child < NBODIES) || work_group_all(distSquared >= dq[depth])) {
							float distance = rsqrt(distSquared);
							float f = _mass[child] * distance * distance * distance;
							//printf("f: %f\n", f);
							accX += distX * f;
							accY += distY * f;
							accZ += distZ * f;
							
						} else {
							depth++;
							
							if (sBase == get_local_id(0)) {
								localNode[depth] = NUMBER_OF_NODES;
								localPos[depth] = 0;
							}
						
							// TODO is this memfence needed?
							mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
						}
					} else {
						depth = max(j, depth -1);
					}
				}
				
				// done with this level
				depth--;
			}
	
			if (*_step > 0) {
				// update velocity
				_velX[sortedIndex] += (accX - _accX[sortedIndex]) * TIMESTEP * 0.5f;
				_velY[sortedIndex] += (accY - _accY[sortedIndex]) * TIMESTEP * 0.5f;
				_velZ[sortedIndex] += (accZ - _accZ[sortedIndex]) * TIMESTEP * 0.5f;
				
				// TODO some sort of barrier?
			}			
			
			_accX[sortedIndex] = accX;
			_accY[sortedIndex] = accY;
			_accZ[sortedIndex] = accZ;
		}
	}
}