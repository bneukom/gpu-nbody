
// __attribute__ ((reqd_work_group_size(10, 1, 1)))
__kernel void test(__global float *out)
{ 
	int globalId = get_global_id(0);
	int localId = get_local_id(0);

	atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_seq_cst, memory_scope_device);
    out[globalId] = out[globalId] + globalId;
}

