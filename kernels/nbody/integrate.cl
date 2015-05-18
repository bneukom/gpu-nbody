#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

#include "kernels/nbody/debug.h"

#define NULL_BODY (-1)
#define LOCK (-2)

// TODO pass as argument
#define MAXDEPTH 64
#define TIMESTEP (0.025f)
//#define TIMESTEP 0.00078125f

#define NUMBER_OF_CELLS 8 // the number of cells per node

__attribute__ ((reqd_work_group_size(WORKGROUP_SIZE, 1, 1)))
__kernel void integrate(
	__global float* _posX, __global float* _posY, __global float* _posZ,
	__global float* _velX, __global float* _velY, __global float* _velZ, 
	__global float* _accX, __global float* _accY, __global float* _accZ,  
	__global int* _step, __global int* _blockCount, __global int* _bodyCount,  __global float* _radius, __global int* _maxDepth,
	__global int* _bottom, __global volatile float* _mass, __global volatile int* _child, __global volatile int* _start, __global volatile int* _sorted, __global int* _error) {
	
	int stepSize = get_local_size(0) * get_num_groups(0);
	
	for (int i = get_global_id(0); i < NBODIES; i += stepSize) {

		float deltaVelX = _accX[i] * TIMESTEP * 0.5f;	
		float deltaVelY = _accY[i] * TIMESTEP * 0.5f;
		float deltaVelZ = _accZ[i] * TIMESTEP * 0.5f;
		
		float velX = _velX[i] + deltaVelX;
		float velY = _velY[i] + deltaVelY;
		float velZ = _velZ[i] + deltaVelZ;
		
		_posX[i] += velX * TIMESTEP;
		_posY[i] += velY * TIMESTEP;
		_posZ[i] += velZ * TIMESTEP;
		
		_velX[i] = velX + deltaVelX;
		_velY[i] = velY + deltaVelY;
		_velZ[i] = velZ + deltaVelZ;
		
		DEBUG_PRINT(("velX[%d]: %f\n", i, _velX[i]));
		DEBUG_PRINT(("velY[%d]: %f\n", i, _velY[i]));
		DEBUG_PRINT(("velZ[%d]: %f\n", i, _velZ[i]));
		
		/*
		_posX[i] += TIMESTEP * _velX[i] + 0.5f * TIMESTEP * TIMESTEP * _accX[i];
		_posY[i] += TIMESTEP * _velY[i] + 0.5f * TIMESTEP * TIMESTEP * _accY[i];
		_posZ[i] += TIMESTEP * _velZ[i] + 0.5f * TIMESTEP * TIMESTEP * _accZ[i];
		
		_velX[i] += TIMESTEP * _accX[i];
		_velY[i] += TIMESTEP * _accY[i];
		_velZ[i] += TIMESTEP * _accZ[i];
		*/
		
		/*
		_velX[i] += TIMESTEP * _accX[i];
		_velY[i] += TIMESTEP * _accY[i];
		_velZ[i] += TIMESTEP * _accZ[i];
		
		_posX[i] += TIMESTEP * _velX[i];
		_posY[i] += TIMESTEP * _velY[i];
		_posZ[i] += TIMESTEP * _velZ[i];
		*/
		
		
	}
}