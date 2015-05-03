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
#define TIMESTEP 0.025f

#define NUMBER_OF_CELLS 8 // the number of cells per node

__attribute__ ((reqd_work_group_size(WORKGROUP_SIZE, 1, 1)))
__kernel void integrate(
	__global float* _posX, __global float* _posY, __global float* _posZ,
	__global float* _velX, __global float* _velY, __global float* _velZ, 
	__global float* _accX, __global float* _accY, __global float* _accZ,  
	__global int* _step, __global int* _blockCount, __global int* _bodyCount,  __global float* _radius, __global int* _maxDepth,
	__global int* _bottom, __global volatile float* _mass, __global volatile int* _child, __global volatile int* _start, __global volatile int* _sorted) {
	
	int stepSize = get_local_size(0) * get_num_groups(0);
	
	for (int i = get_global_id(0); i < NBODIES; i += stepSize) {
		float deltaVelX = _accX[i] * TIMESTEP * 0.5f;	
		float deltaVelY = _accY[i] * TIMESTEP * 0.5f;
		float deltaVelZ = _accZ[i] * TIMESTEP * 0.5f;
		
		float velX = _velX[i] + deltaVelX;
		float velY = _velY[i] + deltaVelY;
		float velZ = _velZ[i] + deltaVelZ;
		
		_posX[i] += velX * TIMESTEP;
		_posX[i] += velX * TIMESTEP;
		_posX[i] += velX * TIMESTEP;
		
		_velX[i] = velX + deltaVelX;
		_velY[i] = velY + deltaVelY;
		_velZ[i] = velZ + deltaVelZ;
	}
}