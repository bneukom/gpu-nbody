_attribute__ ((reqd_work_group_size(THREADS2, 1, 1)))
__kernel void NBODY_KERNEL(buildTree)
{
    __local real radius, rootX, rootY, rootZ;
    __local volatile int successCount;
    __local volatile int doneCount; /* Count of items loaded in the tree */
    __local volatile int deadCount; /* Count of items in workgroup finished */

    const int maxN = HAVE_CONSISTENT_MEMORY ? maxNBody : NBODY;
    int localMaxDepth = 1;
    bool newParticle = true;

    uint inc = get_local_size(0) * get_num_groups(0);
    int i = get_global_id(0);

    if (get_local_id(0) == 0)
    {
        /* Cache root data */
        radius = _treeStatus->radius;
        rootX = _posX[NNODE];
        rootY = _posY[NNODE];
        rootZ = _posZ[NNODE];

        doneCount = _treeStatus->doneCnt;
        successCount = 0;
        deadCount = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    if (HAVE_CONSISTENT_MEMORY)
    {
        (void) atom_add(&deadCount, i >= maxN);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (!HAVE_CONSISTENT_MEMORY)
    {
        if (doneCount == NBODY)
            return;
    }

    cl_assert_rtn(_treeStatus, !isnan(radius) && !isinf(radius));

    /* If we know we have consistent global memory across workgroups,
     * we will continue this loop until we completely construct the
     * tree in a single kernel call, limited by the upper bound
     * on the number of particles for responsiveness.
     *
     * If we don't have consistent memory, we will try once per
     * particle to load it, and if it fails we abandon it. We repeat
     * the kernel until everything is completed. We also need to do
     * additional checking to make sure that all values we depend on
     * are fully visible to the workitem before using them.
     *
     */

  #if HAVE_CONSISTENT_MEMORY
    while (deadCount != THREADS2) /* We need to avoid conditionally barriering when reducing mem. pressure */
  #else
    while (i < maxN)   /* We can just keep going until we are done with no barrier */
  #endif
    {
        if (!HAVE_CONSISTENT_MEMORY || i < maxN)
        {
            real r;
            real px, py, pz;
            int j, n, depth;
            bool posNotReady;

            if (newParticle)
            {
                /* New body, so start traversing at root */
                newParticle = false;
                posNotReady = false;

                px = _posX[i];
                py = _posY[i];
                pz = _posZ[i];
                n = NNODE;
                depth = 1;
                r = radius;

                /* Determine which child to follow */
                j = 0;
                if (rootX <= px)
                    j = 1;
                if (rootY <= py)
                    j |= 2;
                if (rootZ <= pz)
                    j |= 4;
            }

            int ch = _child[NSUB * n + j];
            while (ch >= NBODY && !posNotReady && depth <= MAXDEPTH)  /* Follow path to leaf cell */
            {
                n = ch;
                ++depth;
                r *= 0.5;

                real pnx = _posX[n];
                real pny = _posY[n];
                real pnz = _posZ[n];

                /* Test if we don't have a consistent view. We
                   initialized these all to NAN so we can be sure we
                   have a good view once actually written.

                   This is in case we don't have cross-workgroup global memory consistency
                 */
                posNotReady = isnan(pnx) || isnan(pny) || isnan(pnz);

                /* Determine which child to follow */
                j = 0;
                if (pnx <= px)
                    j = 1;
                if (pny <= py)
                    j |= 2;
                if (pnz <= pz)
                    j |= 4;
                ch = _child[NSUB * n + j];
            }

            /* Skip if child pointer is locked, or the same particle, and try again later.

               If we have consistent memory we only need to check if ch != LOCK.
             */
            if ((ch != LOCK) && (ch != i) && !posNotReady)
            {
                int locked = NSUB * n + j;

                if (ch == atom_cmpxchg(&_child[locked], ch, LOCK)) /* Try to lock */
                {
                    if (ch == -1)
                    {
                        /* If null, just insert the new body */
                        _child[locked] = i;
                    }
                    else  /* There already is a body in this position */
                    {
                        int patch = -1;
                        /* Create new cell(s) and insert the old and new body */
                        do
                        {
                            ++depth;

                            int cell = atom_dec(&_treeStatus->bottom) - 1;
                            if (cell <= NBODY)
                            {
                                _treeStatus->errorCode = NBODY_KERNEL_CELL_OVERFLOW;
                                _treeStatus->bottom = NNODE;
                            }
                            patch = max(patch, cell);

                            if (SW93 || NEWCRITERION)
                            {
                                _critRadii[cell] = r;  /* Save cell size */
                            }

                            real nx = _posX[n];
                            real ny = _posY[n];
                            real nz = _posZ[n];

                            cl_assert(_treeStatus, !isnan(nx) && !isnan(ny) && !isnan(nz));

                            r *= 0.5;

                            real x = nx + (px < nx ? -r : r);
                            real y = ny + (py < ny ? -r : r);
                            real z = nz + (pz < nz ? -r : r);

                            _posX[cell] = x;
                            _posY[cell] = y;
                            _posZ[cell] = z;

                            if (patch != cell)
                            {
                                _child[NSUB * n + j] = cell;
                            }


                            real pchx = _posX[ch];
                            real pchy = _posY[ch];
                            real pchz = _posZ[ch];

                            cl_assert(_treeStatus, !isnan(pchx) && !isnan(pchy) && !isnan(pchz));

                            j = 0;
                            if (x <= pchx)
                                j = 1;
                            if (y <= pchy)
                                j |= 2;
                            if (z <= pchz)
                                j |= 4;

                            _child[NSUB * cell + j] = ch;

                            /* The AMD compiler reorders the next read
                             * from _child, which then reads the old/wrong
                             * value when the children are the same without this.
                             */
                            maybe_strong_global_mem_fence();

                            n = cell;
                            j = 0;
                            if (x <= px)
                                j = 1;
                            if (y <= py)
                                j |= 2;
                            if (z <= pz)
                                j |= 4;

                            ch = _child[NSUB * n + j];

                            /* Repeat until the two bodies are
                             * different children or we overflow */
                        }
                        while (ch >= 0 && depth <= MAXDEPTH);

                        _child[NSUB * n + j] = i;
                        maybe_strong_global_mem_fence();
                        _child[locked] = patch;
                    }
                    maybe_strong_global_mem_fence();

                    localMaxDepth = max(depth, localMaxDepth);
                    if (HAVE_CONSISTENT_MEMORY)
                    {
                        i += inc;  /* Move on to next body */
                        newParticle = true;
                        (void) atom_add(&deadCount, i >= maxN);
                    }
                    else
                    {
                        (void) atom_inc(&successCount);
                    }
                }
            }

            if (!HAVE_CONSISTENT_MEMORY)
            {
                i += inc;  /* Move on to next body */
                newParticle = true;
            }
        }

       if (HAVE_CONSISTENT_MEMORY)
       {
            /* Wait for other wavefronts to finish loading to reduce
             * memory pressures */
           barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
       }
    }

    if (!HAVE_CONSISTENT_MEMORY)
    {
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        if (get_local_id(0) == 0)
        {
            (void) atom_add(&_treeStatus->doneCnt, successCount);
        }
    }

    (void) atom_max(&_treeStatus->maxDepth, localMaxDepth);
}