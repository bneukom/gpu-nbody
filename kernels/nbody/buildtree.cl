#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

//#include "kernels/nbody/debug.h"

#ifdef DEBUG
# define DEBUG_PRINT(x) printf x
#else
# define DEBUG_PRINT(x) do {} while (0)
#endif

#define NULL_BODY (-1)
#define LOCK (-2)

#define NUMBER_OF_CELLS 8 // the number of cells per node

__attribute__ ((reqd_work_group_size(WORKGROUP_SIZE, 1, 1)))
__kernel void buildTree(
	__global float* _posX, __global float* _posY, __global float* _posZ, 
	__global int* _blockCount, __global float* _radius, __global int* _bottom, __global float* _mass, __global int* _child) {

    int localMaxDepth = 1;
	
    const int stepSize = get_local_size(0) * get_num_groups(0);
  	DEBUG_PRINT(("- Info -\n"));
  	DEBUG_PRINT(("NUMBER_OF_NODES: %d\n", NUMBER_OF_NODES));
  	DEBUG_PRINT(("NBODIES: %d\n", NBODIES));
  	
    // Cache root data 
    float radius = *_radius;
    float rootX = _posX[NUMBER_OF_NODES];
	float rootY = _posY[NUMBER_OF_NODES];
    float rootZ = _posZ[NUMBER_OF_NODES];

	DEBUG_PRINT(("rootX: %f\n", rootX));
	DEBUG_PRINT(("rootY: %f\n", rootY));
	DEBUG_PRINT(("rootZ: %f\n", rootZ));
	DEBUG_PRINT(("radius: %f\n", *_radius));
	int childPath;

	bool newBody = true;
	int node;
	
	// iterate over all bodies assigned to this thread
	int bodyIndex = get_global_id(0);
	DEBUG_PRINT(("- Iterate over Bodies -\n"));
    while (bodyIndex < NBODIES) {
    	DEBUG_PRINT(("bodyIndex: %d\n", bodyIndex));
    	DEBUG_PRINT(("\tnew body: %s\n", newBody ? "true" : "false"));
        float currentR;
        float bodyX, bodyY, bodyZ;
        int depth;

        if (newBody) {
			// new body, so start traversing at root
            newBody = false;

            bodyX = _posX[bodyIndex];
            bodyY = _posY[bodyIndex];
            bodyZ = _posZ[bodyIndex];
            DEBUG_PRINT(("\tbodyX: (%f, %f, %f)\n", bodyX, bodyY, bodyZ));
	
            node = NUMBER_OF_NODES;
            depth = 1;
            currentR = radius;

            // Determine which child to follow
            childPath = 0;
            if (rootX < bodyX) childPath  = 1;
            if (rootY < bodyY) childPath += 2;
            if (rootZ < bodyZ) childPath += 4;
            DEBUG_PRINT(("\tchildPath: %d\n", childPath));
        }

        int childIndex = _child[NUMBER_OF_CELLS * node + childPath];
		DEBUG_PRINT(("\tchildIndex: %d\n", childIndex));
		DEBUG_PRINT(("\tfind leaf cell: %d\n", childIndex));
		// follow path to leaf cell
        while (childIndex >= NBODIES) {
            node = childIndex;
            ++depth;
            currentR *= 0.5f;

			// Determine which child to follow
            childPath = 0;
            if (_posX[node] < bodyX) childPath  = 1;
            if (_posY[node] < bodyY) childPath += 2;
            if (_posZ[node] < bodyZ) childPath += 4;

            childIndex = _child[NUMBER_OF_CELLS * node + childPath];
        }
        
        DEBUG_PRINT(("\tleaf cell childIndex: %d\n", childIndex));

        if (childIndex != LOCK) { 
            int locked = NUMBER_OF_CELLS * node + childPath;
            DEBUG_PRINT(("\tlock: %d\n", locked));
            DEBUG_PRINT(("\tnode: %d\n", node));
            if (childIndex == atom_cmpxchg(&_child[locked], childIndex, LOCK)) { // try locking
                if (childIndex == NULL_BODY) {
					// no body has been here, so just insert
					DEBUG_PRINT(("\t -> no body in cell - insert body %d at %d\n", bodyIndex, locked));
                    _child[locked] = bodyIndex;
                } else {
                	DEBUG_PRINT(("\t -> childIndex %d already used\n", childIndex));
                    int patch = -1;
                    // Create new cell(s) and insert the old and new body
                    do {
                        depth++;
						DEBUG_PRINT(("\t\titeration %d >= 0\n", childIndex));
                        const int cell = atom_dec(_bottom) - 1;
                        DEBUG_PRINT(("\t\tcell %d\n", cell));
                         DEBUG_PRINT(("\t\tnode: %d\n", node));
                        if (cell <= NBODIES)  {
							// TODO REPORT ERROR
                          	printf("ERROR ABORT\n");
                          	*_bottom = NUMBER_OF_NODES;
                          	return;
                        }
                        patch = max(patch, cell);

						DEBUG_PRINT(("\t\tchildPath & 1: %d\n", childPath & 1));
						float x = (childPath & 1) * currentR;
						float y = ((childPath >> 1) & 1) * currentR;
						float z = ((childPath >> 2) & 1) * currentR;
              
                        currentR *= 0.5f;
						
						_mass[cell] = -1.0f;
						// TODO StartD ???
                        x = _posX[cell] = _posX[node] - currentR + x;
                        y = _posY[cell] = _posY[node] - currentR + y;
                        z = _posZ[cell] = _posZ[node] - currentR + z;

						#pragma unroll NUMBER_OF_CELLS
						for (int k = 0; k < NUMBER_OF_CELLS; k++) _child[cell * NUMBER_OF_CELLS + k] = -1;

                        if (patch != cell) {
							DEBUG_PRINT(("\t\tinsert cell %d at %d\n", cell, NUMBER_OF_CELLS * node + childPath));
                            DEBUG_PRINT(("\t\t\tnode: %d\n", node));
                            DEBUG_PRINT(("\t\t\tchildpath: %d\n", childPath));
                            _child[NUMBER_OF_CELLS * node + childPath] = cell;
                        }


                        childPath = 0;
                        if (x < _posX[childIndex]) childPath = 1;
                        if (y < _posY[childIndex]) childPath += 2;         
                        if (z < _posZ[childIndex]) childPath += 4;
                        _child[NUMBER_OF_CELLS * cell + childPath] = childIndex;
						
						// next child
                        node = cell;
                        childPath = 0;
                        if (x < bodyX) childPath = 1;             
                        if (y < bodyY) childPath += 2;
                        if (z < bodyZ) childPath += 4;
                            
                        childIndex = _child[NUMBER_OF_CELLS * node + childPath];
                        DEBUG_PRINT(("\t\tchildIndex: %d\n", childIndex));
                        DEBUG_PRINT(("\t\tnode: %d\n", node));
                        DEBUG_PRINT(("\t\t=======\n"));
                    } while (childIndex >= 0);
                    
                    DEBUG_PRINT(("\tadded subtree push out\n"));
                    DEBUG_PRINT(("\tinsert body %d at %d\n", bodyIndex, NUMBER_OF_CELLS * node + childPath));
                    _child[NUMBER_OF_CELLS * node + childPath] = bodyIndex;

					// push out
					atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_seq_cst, memory_scope_device);
		
                    _child[locked] = patch;
                }
                
				atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_seq_cst, memory_scope_device);
		
                localMaxDepth = max(depth, localMaxDepth);

				// move to next body
                bodyIndex += stepSize;
                newBody = true;           
            }
		}

		// wait for others to finish to reducue memory pressure
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
		
    }

	// TODO MAX DEPTH
    // (void) atom_max(_maxDepth, localMaxDepth);
}