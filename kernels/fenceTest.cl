__kernel void fenceTest(__global int *out)
{ 
	int id = get_global_id(0);
 
    // Every thread finishes the atomic op at different times
    atom_add(scratch, id);
 
    // Do some work dependent on each workgroup ...
    tmp[id] = get_group_id(0);

    // ... do some work that depends on a different workgroup
    out[id] = tmp[get_global_size(0) - id - 1];
}

