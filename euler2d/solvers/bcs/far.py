import numba as nb
import numpy as np


def make_bc_far(nvars=4, rhob=1.0, ub=1.0, vb=0.0, pb=1.0, gamma=1.4):

    # Speed of sound entropy at bc
    ab = np.sqrt(gamma*pb / rhob)
    sb = pb / rhob**gamma
    ab_gmo = 2*ab/(gamma-1)
    
    @nb.jit(nopython=True)
    def _bc(ul, ur, nf):
        # Implement Far BC!!!
        
        # 경계 속도
        qb = ub*nf[0] + vb*nf[1]
        
        # 경계 Riemann invariants
        Rp_b = qb + ab_gmo
        Rm_b = qb - ab_gmo
        
        # 내부 격자 primitive variables
        rhoL = ul[0]
        uL = ul[1] / ul[0]
        vL = ul[2] / ul[0]
        pL = (gamma-1) * (ul[3] - 0.5*rhoL*(uL**2 + vL**2))
        aL =  np.sqrt(gamma*pL/rhoL)
        sL = pL / rhoL**gamma
        aL_gmo = 2*aL/(gamma-1)
        
        # 내부 격자 속도  
        qL = uL*nf[0] + vL*nf[1]
        
        # 내부 격자 Riemann invariants
        Rp_L = qL + aL_gmo
        Rm_L = qL - aL_gmo
        
        # Inflow
        if qb < 0:
            # Subsonic
            if abs(qb) < ab:
                Rp = Rp_L
                Rm = Rm_b
                V = qb
                s = sb
                
            # Supersonic
            else:
                Rp = Rp_b
                Rm = Rm_b
                V = qb
                s = sb
                
        # Outflow        
        else:
            # Subsonic
            if abs(qb) < ab: 
                Rp = Rp_L
                Rm = Rm_b
                V = qL
                s = sL
            # Supersonic 
            else:
                Rp = Rp_L
                Rm = Rm_L
                V = qL
                s = sL
                
        # Riemann invariants(R+,R-)를 통한 ghost cell의 물성치
        U = 0.5 * (Rp + Rm)
        c = (gamma-1) * (Rp-Rm) / 4
        rho = (c**2/(gamma*s)) ** (1/(gamma-1))
        p = s * rho**gamma
                

        ug = ub + (U-V)*nf[0]
        vg = vb + (U-V)*nf[1]
        
        ur[0] = rho
        ur[1] = rho * ug
        ur[2] = rho * vg
        ur[3] = p/(gamma-1) + 0.5*rho*(ug**2+vg**2)
           
    return _bc
