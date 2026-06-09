import numpy as np
from solvers.flux import to_flux

def make_stegarwarming(gamma=1.4, nvars=3):
    
    fl, fr = np.empty(nvars), np.empty(nvars)
    
    def _flux(ul, ur, fn):
        # fl, fr = np.empty(nvars), np.empty(nvars) 
        # Variables
        pl, vl = to_flux(ul, fl)
        pr, vr = to_flux(ur, fr)
        rhol, rhor = ul[0], ur[0]
        el, er = ul[2], ur[2]
        hl, hr = (el+pl) / rhol, (er+pr) / rhor
        al, ar = np.sqrt(gamma*pl/rhol), np.sqrt(gamma*pr/rhor)
        # Ml, Mr = vl/al, vr/ar
        
        # Eigenvalues (u-a, u, u+a)
        lam1l, lam1r = vl-al, vr-ar
        lam2l, lam2r = vl, vr
        lam3l, lam3r = vl+al, vr+ar
        
        # Splitting of the eigenvalues(+,-)
        lam1lp = 0.5*(lam1l+abs(lam1l))
        lam1rm = 0.5*(lam1r-abs(lam1r))
        lam2lp = 0.5*(lam2l+abs(lam2l))
        lam2rm = 0.5*(lam2r-abs(lam2r))
        lam3lp = 0.5*(lam3l+abs(lam3l))
        lam3rm = 0.5*(lam3r-abs(lam3r))
        
        # Split flux compononet
        # flp, frm = np.empty(nvars), np.empty(nvars)
        flp = np.array([(rhol/2*gamma)*(lam1lp + 2*(gamma-1)*lam2lp + lam3lp),
                       (rhol/2*gamma)*((vl-al)*lam1lp + 2*(gamma-1)*vl*lam2lp + (vl+al)*lam3lp),
                       (rhol/2*gamma)*((hl-vl*al)*lam1lp + (gamma-1)*vl**2*lam2lp + (hl+vl*al)*lam3lp)])
        frm = np.array([(rhor/2*gamma)*(lam1rm + 2*(gamma-1)*lam2rm + lam3rm),
                       (rhor/2*gamma)*((vr-ar)*lam1rm + 2*(gamma-1)*vr*lam2rm + (vr+ar)*lam3rm),
                       (rhor/2*gamma)*((hr-vr*ar)*lam1rm + (gamma-1)*vr**2*lam2rm + (hr+vr*ar)*lam3rm)])
        for jdx in range(nvars):
            fn[jdx] = flp[jdx] + frm[jdx]
        
    return _flux
