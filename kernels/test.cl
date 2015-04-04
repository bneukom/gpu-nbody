__kernel void test(__global int *out)
{ 
	int globalId = get_global_id(0);
	int localId = get_local_id(0);

	atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_seq_cst, memory_scope_device);
    out[globalId] = globalId;
}

