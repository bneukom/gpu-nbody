// TODO better adopt http://developer.amd.com/resources/documentation-articles/articles-whitepapers/opencl-optimization-case-study-simple-reductions/
// TODO for small number of bodies does not quite work!

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

#include "kernels/nbody/debug.h"

/*
 Calculates the bounding box around all bodies.
*/
__attribute__ ((reqd_work_group_size(BOUNDING_BOX_WORKGROUP_SIZE, 1, 1))) 
__kernel void boundingBox(
	__global float* _posX, __global float* _posY, __global float* _posZ, 
	__global volatile float* _minX, __global volatile float* _minY, __global volatile float* _minZ, 
	__global volatile float* _maxX, __global volatile float* _maxY, __global volatile float* _maxZ, 
	__global int* _blockCount, __) {

	DEBUG_PRINT(("INFO\n"));
	DEBUG_PRINT(("block_count = %d\n", *_blockCount));
	float minX, maxX, minY, maxY, minZ, maxZ;
    __local volatile float localMinX[BOUNDING_BOX_WORKGROUP_SIZE], localMinY[BOUNDING_BOX_WORKGROUP_SIZE], localMinZ[BOUNDING_BOX_WORKGROUP_SIZE];
    __local volatile float localMaxX[BOUNDING_BOX_WORKGROUP_SIZE], localMaxY[BOUNDING_BOX_WORKGROUP_SIZE], localMaxZ[BOUNDING_BOX_WORKGROUP_SIZE];

    const int localId = get_local_id(0);
	const int groupId = get_group_id(0);

	DEBUG_PRINT(("get_num_groups = %d\n", get_num_groups(0)));
	DEBUG_PRINT(("get_local_size = %d\n", get_local_size(0)));
	DEBUG_PRINT(("group_id = %d\n", groupId));
	DEBUG_PRINT(("local_id = %d\n", localId));

	// TODO use infinity?
	// initialize with valid data (in case #bodies < #threads)
	minX = maxX = _posX[0];
	minY = maxY = _posY[0];
	minZ = maxZ = _posZ[0];

	// scan all bodies
	// TODO is this get_global_size() ???
    int stepSize = get_local_size(0) * get_num_groups(0); // get_local_size(0) = NUM_THREADS 
	DEBUG_PRINT(("stepSize = %d\n", stepSize));
	DEBUG_PRINT(("nbodies = %d\n", NBODIES));
	DEBUG_PRINT(("CODE\n"));
	for (int i = localId + get_group_id(0) * get_local_size(0); i < NBODIES; i += stepSize) { 
		DEBUG_PRINT(("i = %d (%d, %d)\n", i, get_group_id(0), get_local_id(0)));
	    float pos = _posX[i];
		DEBUG_PRINT(("particle x %f (%d, %d)\n", pos, get_group_id(0), get_local_id(0)));
        minX = fmin(minX, pos);
		DEBUG_PRINT(("minX: %f (%d, %d)\n", minX, get_group_id(0), get_local_id(0)));
        maxX = fmax(maxX, pos);
		DEBUG_PRINT(("maxX: %f (%d, %d)\n", maxX, get_group_id(0), get_local_id(0)));
		 
        pos = _posY[i];
		DEBUG_PRINT(("particle y %f (%d, %d)\n", pos, get_group_id(0), get_local_id(0)));
        minY = fmin(minY, pos);
		DEBUG_PRINT(("minY: %f (%d, %d)\n", minY, get_group_id(0), get_local_id(0)));
        maxY = fmax(maxY, pos);
		DEBUG_PRINT(("maxY: %f (%d, %d)\n", maxY, get_group_id(0), get_local_id(0)));

        pos = _posZ[i];
		DEBUG_PRINT(("particle z %f (%d, %d)\n", pos, get_group_id(0), get_local_id(0)));
        minZ = fmin(minZ, pos);
		DEBUG_PRINT(("minZ: %f (%d, %d)\n", minZ, get_group_id(0), get_local_id(0)));
        maxZ = fmax(maxZ, pos);
		DEBUG_PRINT(("maxZ: %f (%d, %d)\n", maxZ, get_group_id(0), get_local_id(0)));
	}

	for (int offset  = get_local_size(0) / 2; offset > 0; offset  /= 2) {
		DEBUG_PRINT(("offset: %d (%d, %d)\n", offset, get_group_id(0), get_local_id(0))); 
		barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

        if (localId < offset) {
			
            localMinX[localId] = minX = fmin(minX, localMinX[localId + offset]);
            localMinY[localId] = minY = fmin(minY, localMinY[localId + offset]);
            localMinZ[localId] = minZ = fmin(minZ, localMinZ[localId + offset]);

            localMaxX[localId] = maxX = fmax(maxX, localMaxX[localId + offset]);
            localMaxY[localId] = maxY = fmax(maxY, localMaxY[localId + offset]);
            localMaxZ[localId] = maxZ = fmax(maxZ, localMaxZ[localId + offset]);
        }
    }

	
	// Write block result to global memory
    if (localId == 0) {
		DEBUG_PRINT(("Result from Group\n"));
		DEBUG_PRINT(("minX: %f\n", minX));
		DEBUG_PRINT(("minY: %f\n", minY));
		DEBUG_PRINT(("minZ: %f\n", minZ));
		DEBUG_PRINT(("maxX: %f\n", maxX));
		DEBUG_PRINT(("maxY: %f\n", maxY));
		DEBUG_PRINT(("maxZ: %f\n", maxZ));
        _minX[groupId] = minX;
        _minY[groupId] = minY;
        _minZ[groupId] = minZ;

        _maxX[groupId] = maxX;
        _maxY[groupId] = maxY;
        _maxZ[groupId] = maxZ;

		// wait for completion
		mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
		atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_seq_cst, memory_scope_device);

		
        const int lastBlockId = get_num_groups(0) - 1;
        if (lastBlockId == atom_inc(_blockCount)) {
			
            // last block so combine all results
            for (int i = 0; i <= lastBlockId; ++i) {
                minX = fmin(minX, _minX[i]);
                minY = fmin(minY, _minY[i]);
                minZ = fmin(minZ, _minZ[i]);

                maxX = fmax(maxX, _maxX[i]);
                maxY = fmax(maxY, _maxY[i]);
                maxZ = fmax(maxZ, _maxZ[i]);
            }

			DEBUG_PRINT(("Result\n"));
			DEBUG_PRINT(("blocks: %d\n", *_blockCount));
			DEBUG_PRINT(("minX: %f\n", minX));
			DEBUG_PRINT(("minY: %f\n", minY));
			DEBUG_PRINT(("minZ: %f\n", minZ));
			DEBUG_PRINT(("maxX: %f\n", maxX));
			DEBUG_PRINT(("maxY: %f\n", maxY));
			DEBUG_PRINT(("maxZ: %f\n", maxZ));
        }
		
    }
}

