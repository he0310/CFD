from nbutils import array
from solvers.fluid import to_primitive
from solvers.flux_jac import to_flux_pm_jac

import numba as nb
import numpy as np


def make_lusgs(nfx, nfy, rhside, jac, si, sj, kappa=1.02, gamma=1.4, nvars=4, npads=(1,1)):
    # Temporal arrays
    du = np.zeros((nvars, nfy+2*npads[1], nfx+2*npads[0]))
    dut = np.zeros_like(du)
    rhs = np.zeros((nvars, nfy+2*npads[1], nfx+2*npads[0]))
    
    # Temporal arrays for flux
    ff = np.zeros((nvars, nfy, nfx+1))
    fg = np.zeros((nvars, nfy+1, nfx))
    
    # Diagonal matrix
    diag = np.zeros((nfy, nfx))
    
    # Metric at the cell
    sic = 0.5*(si[:, :, :-1] + si[:, :, 1:])*jac
    sjc = 0.5*(sj[:, :-1, :] + sj[:, 1:, :])*jac

    # Magnitude of the metric
    aic = np.linalg.norm(sic, axis=0)
    ajc = np.linalg.norm(sjc, axis=0)
    
    # Normalize (Normal vector)
    sic /= aic
    sjc /= ajc
    
    @nb.jit(nopython=True)
    def _lhs(dt, q, diag):
        for j in range(nfy):
            jy = j + npads[1]
            for i in range(nfx):
                ix = i + npads[0]
                qp = array((nvars,))
                
                to_primitive(q[:, jy, ix], qp)
                rho, u, v, p = qp
                a = np.sqrt(gamma*p/rho)
                
                # Spectral radius
                ra = (abs(sic[0, j, i]*u + sic[1, j, i]*v) + a)*aic[j, i]
                rb = (abs(sjc[0, j, i]*u + sjc[1, j, i]*v) + a)*ajc[j, i]
                
                diag[j, i] = 1 / dt[jy, ix] + kappa*(ra + rb)
                    
    @nb.jit(nopython=True)
    def _lsweep(q, du, dut):
        for j in range(nfy):
            jy = j + npads[1]
            for i in range(nfx):
                ix = i + npads[0]

                res = np.zeros((nvars,))
                jac = np.zeros((nvars, nvars))

                # i-1 face
                to_flux_pm_jac(q[:, jy, ix-1], jac, sic[:, j, i-1], +1.0)
                res += jac * dut[:, jy, ix-1]

                # j-1 face
                to_flux_pm_jac(q[:, jy-1, ix], jac, sjc[:, j-1, i], +1.0)
                res += jac * dut[:, jy-1, ix]

                # Update dut
                dut[:, jy, ix] = (-du[:, jy, ix] + res) / diag[j, i]
                
    @nb.jit(nopython=True)
    def _usweep(q, dut, du):
        for j in range(nfy-1, -1, -1):
            jy = j + npads[1]
            for i in range(nfx-1, -1, -1):
                ix = i + npads[0]

                res = np.zeros((nvars,))
                jac = np.zeros((nvars, nvars))

                # i+1 face
                to_flux_pm_jac(q[:, jy, ix+1], jac, sic[:, j, i], -1.0)
                res += jac * du[:, jy, ix+1]

                # j+1 face
                to_flux_pm_jac(q[:, jy+1, ix], jac, sjc[:, j, i], -1.0)
                res += jac * du[:, jy+1, ix]

                # Final update
                du[:, jy, ix] = dut[:, jy, ix] - res / diag[j, i]
    
    def _step(dt, u):
        rhside(u, rhs, ff, fg)
        
        _lhs(dt, u, diag)
        
        _lsweep(u, rhs, dut)
        _usweep(u, dut, du)

        u += du
        
        # Density residual
        return np.linalg.norm(du[0])

    return _step
