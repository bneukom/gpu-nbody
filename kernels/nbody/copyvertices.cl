#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

#include "kernels/nbody/debug.h"

__attribute__ ((reqd_work_group_size(WORKGROUP_SIZE, 1, 1)))
__kernel void copyVertices(
	__global float* _posX, __global float* _posY, __global float* _posZ, __global float4* pos,
	__global float* _velX, __global float* _velY, __global float* _velZ, __global float4* vel) {
	
	int stepSize = get_local_size(0) * get_num_groups(0);
	
	for (int i = get_global_id(0); i < NBODIES; i += stepSize) {
		pos[i] = (float4)(_posX[i], _posY[i], _posZ[i], 1);
		vel[i] = (float4)(_velX[i], _velY[i], _velZ[i], 1);
	}
}