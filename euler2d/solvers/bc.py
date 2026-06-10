from nbutils import array
from solvers.bcs import get_bc

import numba as nb
import numpy as np


def make_bc(nfx, nfy, si, sj, bctypes, nvars=4, npads=(1,1)):
    """
    BC kernel genertor

    Parameter
    ---------
    nfx - integer
        xi 방향 격자 개수
    nfy - integer
        eta 방향 격자 개수
    si : array
        xi 방향 Face metric
    sj : array
        eta 방향 Face metric
    bctypes : dict
        경계조건 Dictionary
    nvars : integer
        벡터 크기 (기본값: 4)
    npads : integer
        경계를 위한 padding 개수 (기본값: (1,1))
    """
    # Compute unit normal vector
    ni = si / np.linalg.norm(si, axis=0)
    nj = sj / np.linalg.norm(sj, axis=0)    
    
    # Generate bc sweep kernel
    if bctypes['imin'][0] == 'periodic':
        bc_imin = make_bc_i_periodic(nfx+npads[0]-1, npads[0]-1, nfy, -1, nvars, npads)
        bc_imax = make_bc_i_periodic(npads[0], nfx+npads[0], nfy, +1, nvars, npads)
    else:
        bc_imin = make_bc_i(npads[0], npads[0]-1, nfy, bctypes['imin'], ni, nvars, npads)
        bc_imax = make_bc_i(nfx+npads[0]-1, nfx+npads[0], nfy, bctypes['imax'], ni, nvars, npads)
        
    if bctypes['jmin'][0] == 'periodic':
        bc_jmin = make_bc_j_periodic(nfy+npads[1] - 1, npads[1]-1, nfx, -1, nvars, npads)
        bc_jmax = make_bc_j_periodic(npads[1], nfy+npads[1], nfx, +1, nvars, npads)
    else:
        bc_jmin = make_bc_j(npads[1], npads[1]-1, nfx, bctypes['jmin'], nj, nvars, npads)
        bc_jmax = make_bc_j(nfy+npads[1]-1, nfy+npads[1], nfx, bctypes['jmax'], nj, nvars, npads)
    
    @nb.jit(nopython=True)
    def _run(u):
        bc_imin(u)
        bc_imax(u)
        bc_jmin(u)
        bc_jmax(u)
        
    return _run


def make_bc_i(i, ip, nfy, bcinfo, nf, nvars, npads=(1,1)):    
    """
    i 방향 BC Sweep 커널 생성

    Parameter
    ---------
    i - integer
        Innder index
    ip - integer
        Ghost cell index
    nfy - integer
        eta 방향 격자 개수
    bcinfo - turple
        경계조건 이름과 조건 (dict)
    nf - array
        수직벡터
    nvars : integer
        벡터 크기 (기본값: 4)
    npads : integer
        경계를 위한 padding 개수 (기본값: (1,1))
    """
    bcf = get_bc(bcinfo[0], nvars=nvars, **bcinfo[1])
        
    @nb.jit(nopython=True)
    def _run(u):               
        ur = array((nvars,))
        
        for j in range(npads[1], npads[1]+nfy):
            bcf(u[:, j, i], ur, nf[:, j-npads[1], i-npads[0]])
            for k in range(nvars):
                u[k, j, ip] = ur[k]
                
    return _run
                
    
def make_bc_j(j, jp, nfx, bcinfo, nf, nvars, npads=(1,1)):
    """
    j 방향 BC Sweep 커널 생성

    Parameter
    ---------
    j - integer
        Innder index
    jp - integer
        Ghost cell index
    nfx - integer
        xi 방향 격자 개수
    bcinfo - turple
        경계조건 이름과 조건 (dict)
    nf - array
        수직벡터
    nvars : integer
        벡터 크기 (기본값: 4)
    npads : integer
        경계를 위한 padding 개수 (기본값: (1,1))
    """
    bcf = get_bc(bcinfo[0], nvars=nvars, **bcinfo[1])
    
    @nb.jit(nopython=True)
    def _run(u):
        ur = array((nvars,))
        
        for i in range(npads[0], npads[0]+nfx):
            bcf(u[:, j, i], ur, nf[:, j-npads[1], i-npads[0]])
            for k in range(nvars):
                u[k, jp, i] = ur[k]
                    
    return _run


def make_bc_i_periodic(i, ip, nfy, sign, nvars, npads=(1,1)):
    """
    i 방향 대칭 BC 커널 생성

    Parameter
    ---------
    i - integer
        Innder index
    ip - integer
        Ghost cell index
    nfy - integer
        eta 방향 격자 개수
    sign - integer
        방향 (-1: min, +1: max)
    nvars : integer
        벡터 크기 (기본값: 4)
    npad : integer
        i 방향 경계를 위한 padding 개수 (기본값: 1)
    """
    @nb.jit(nopython=True)
    def _run(u):
        for k in range(nvars):
            for j in range(npads[1], npads[1]+nfy):
                for ix in range(npads[0]):
                    u[k, j, ip + sign*ix] = u[k, j, i+ sign*ix]

    return _run


def make_bc_j_periodic(j, jp, nfx, sign, nvars, npads=(1,1)):
    """
    j 방향 대칭 BC 커널 생성

    Parameter
    ---------
    j - integer
        Innder index
    jp - integer
        Ghost cell index
    nfx - integer
        xi 방향 격자 개수
    sign - integer
        방향 (-1: min, +1: max)
    nvars : integer
        벡터 크기 (기본값: 4)
    npad : integer
        j 방향 경계를 위한 padding 개수 (기본값: 1)
    """
    @nb.jit(nopython=True)
    def _run(u):
        for k in range(nvars):
            for jx in range(npads[1]):
                for i in range(npads[0], npads[0]+nfx):                
                    u[k, jp + sign*jx, i] = u[k, j+ sign*jx, i ]
                
    return _run
