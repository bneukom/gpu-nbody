__attribute__ ((reqd_work_group_size(THREADS1, 1, 1)))
__kernel void boundingBox() {
    __local volatile real minX[THREADS1], minY[THREADS1], minZ[THREADS1];
    __local volatile real maxX[THREADS1], maxY[THREADS1], maxZ[THREADS1];

    uint localId = (uint) get_local_id(0);
    if (localId == 0) {
        minX[0] = _posX[0];
        minY[0] = _posY[0];
        minZ[0] = _posZ[0];
    }

    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

    /* initialize with valid data (in case #bodies < #threads) */
    minX[localId] = maxX[localId] = minX[0];
    minY[localId] = maxY[localId] = minY[0];
    minZ[localId] = maxZ[localId] = minZ[0];

    uint inc = get_local_size(0) * get_num_groups(0);
    uint j = i + get_group_id(0) * get_local_size(0); // = get_global_id(0) (- get_global_offset(0))
    while (j < NBODY) /* Scan bodies */
    {
        real tmp = _posX[j];
        minX[i] = fmin(minX[i], tmp);
        maxX[i] = fmax(maxX[i], tmp);

        tmp = _posY[j];
        minY[i] = fmin(minY[i], tmp);
        maxY[i] = fmax(maxY[i], tmp);

        tmp = _posZ[j];
        minZ[i] = fmin(minZ[i], tmp);
        maxZ[i] = fmax(maxZ[i], tmp);

        j += inc;  /* Move on to next body */
    }

    /* Reduction in shared memory */
    j = get_local_size(0) >> 1;
    while (j > 0)
    {
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
        if (i < j)
        {
            minX[i] = fmin(minX[i], minX[i + j]);
            minY[i] = fmin(minY[i], minY[i + j]);
            minZ[i] = fmin(minZ[i], minZ[i + j]);

            maxX[i] = fmax(maxX[i], maxX[i + j]);
            maxY[i] = fmax(maxY[i], maxY[i + j]);
            maxZ[i] = fmax(maxZ[i], maxZ[i + j]);
        }

        j >>= 1;
    }

    if (i == 0)
    {
        /* Write block result to global memory */
        j = get_group_id(0);

        _minX[j] = minX[0];
        _minY[j] = minY[0];
        _minZ[j] = minZ[0];

        _maxX[j] = maxX[0];
        _maxY[j] = maxY[0];
        _maxZ[j] = maxZ[0];
        mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

        inc = get_num_groups(0) - 1;
        if (inc == atom_inc(&_treeStatus->blkCnt))
        {
            /* I'm the last block, so combine all block results */
            for (j = 0; j <= inc; ++j)
            {
                minX[0] = fmin(minX[0], _minX[j]);
                minY[0] = fmin(minY[0], _minY[j]);
                minZ[0] = fmin(minZ[0], _minZ[j]);

                maxX[0] = fmax(maxX[0], _maxX[j]);
                maxY[0] = fmax(maxY[0], _maxY[j]);
                maxZ[0] = fmax(maxZ[0], _maxZ[j]);
            }

            /* Compute radius */
            real tmpR = fmax(maxX[0] - minX[0], maxY[0] - minY[0]);
            real radius = 0.5 * fmax(tmpR, maxZ[0] - minZ[0]);

            real rootX = 0.5 * (minX[0] + maxX[0]);
            real rootY = 0.5 * (minY[0] + maxY[0]);
            real rootZ = 0.5 * (minZ[0] + maxZ[0]);

            _treeStatus->radius = radius;

            _treeStatus->bottom = NNODE;
            _treeStatus->blkCnt = 0;  /* If this isn't 0'd for next time, everything explodes */
            _treeStatus->doneCnt = 0;

            /* Create root node */
            _mass[NNODE] = -1.0;
            _start[NNODE] = 0;
            _posX[NNODE] = rootX;
            _posY[NNODE] = rootY;
            _posZ[NNODE] = rootZ;

            #pragma unroll NSUB
            for (uint k = 0; k < NSUB; ++k)
            {
                _child[NSUB * NNODE + k] = -1;
            }
        }
    }
}


