from nbutils import array
from solvers.bc import make_bc_zero
from solvers.rsolver.get_rsolver import get_rsolver
from solvers.limiter import get_limiter

import numba as nb
import numpy as np

# Very small number to prevent zero division
eps = 1e-12

def make_rhside(nfx, dx, flux, bc, nvars=3, npad=1):    
    """
    Kernal generator for right hand side
    
    Parameters
    ----------
    nfx : integer
        격자 개수
    dx : float
        격자 간격
    flux : function
        근사 리만 해석자 함수
    bc : function
        경계조건 함수
    nvars : integer
        벡터 크기 (기본값: 3)
    npad : integer
        경계를 위한 padding 개수 (기본값: 1)
    """
    fn = np.empty(nvars)
    
    def _run(u, du, f):
        # Compute BC
        bc(u)

        # Compute flux at each face
        for i in range(nfx+1):
            ul = u[:, i + npad -1]
            ur = u[:, i + npad]
            flux(ul, ur, fn)
            f[:, i] = fn

        # negative derivative of flux at each cell
        for i in range(nfx):        
            du[:, npad+i] = -(f[:, i+1] - f[:, i])/dx
            
    return _run


def make_rhside_muscl(nfx, dx, fluxn, bcn='zero', limitern='none', nvars=3, gamma=1.4, npad=1):    
    """
    Kernal generator for right hand side (MUSCL version)
    
    Parameters
    ----------
    nfx : integer
        격자 개수
    dx : float
        격자 간격
    fluxn : string
        근사 리만 해석자 함수 이름
    bcn : string
        경계조건 함수 이름
    limitern : string
        Limiter 이름
    nvars : integer
        벡터 크기 (기본값: 3)
    gamma : float
        비열비 (기본값: 1.4)
    npad : integer
        경계를 위한 padding 개수 (기본값: 1)
    """
    # Generate bc zero (TODO)
    bc = make_bc_zero(nfx, npad)
    
    # Generate flux and phi functions
    flux = get_rsolver(fluxn, gamma, nvars)
    phi = get_limiter(limitern)

    @nb.jit(nopython=True)
    def _run(u, du, f):
        # Allocate local static array
        fn = np.empty(nvars)
        ul, ur = np.empty(nvars), np.empty(nvars)
        # Compute BC
        bc(u)

        # Compute flux at each face
        for i in range(nfx+1):
            # left index i at the face i+1/2
            idx = i + npad -1
            
            if (i == 0) or (i == nfx):
                for j in range(nvars):
                    ul[j] = u[j, idx]
                    ur[j] = u[j, idx+1]
            else:                
                for j in range(nvars):
                    dp = u[j, idx+1] - u[j, idx]
                    dm = u[j, idx] - u[j, idx-1]

                    sl = phi(dp, dm) * dm
                    ul[j] = u[j, idx] + 0.5 * sl

                    dp = u[j, idx+2] - u[j, idx+1]
                    dm = u[j, idx+1] - u[j, idx]

                    sr = phi(dp, dm) * dm
                    ur[j] = u[j, idx+1] - 0.5*sr
            
            flux(ul, ur, fn)
            f[:, i] = fn

        # negative derivative of flux at each cell
        for i in range(nfx):        
            du[:, npad+i] = -(f[:, i+1] - f[:, i])/dx
            
    return _run
