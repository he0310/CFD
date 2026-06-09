import numpy as np
from nbutils import array
from solvers.flux import to_flux


def rusanov(gamma=1.4, nvars=3):
    """
    Rusanov flux 계산 함수 생성
    
    Parameters
    ----------
    gamma : float
        Specific heat ratio (기본값: 1.4)
    nvars : integer
        벡터 크기 (기본값: 3)
        
    Return
    ------
    _flux : function
        Rusanov flux 함수
    """
    fl, fr = np.empty(nvars), np.empty(nvars)
    
    def _flux(ul, ur, fn):
        pl, vl = to_flux(ul, fl)
        pr, vr = to_flux(ur, fr)
        
        vn = 0.5*(vl + vr)
        an = np.sqrt(gamma*(pl+pr) / (ul[0] + ur[0])) + np.abs(vn)
        
        for jdx in range(nvars):
            fn[jdx] = 0.5*(fl[jdx] + fr[jdx]) - 0.5*an*(ur[jdx] - ul[jdx])
            
    return _flux

# Flux vector splitting
# Stegar-Warming scheme
def stegarwarming(gamma=1.4, nvars=3):
    fl, fr = np.empty(nvars), np.empty(nvars)
    
    def _flux(ul, ur, fn):
        flp, frm = np.empty(nvars), np.empty(nvars)
        
        # Variables
        pl, vl = to_flux(ul, fl)
        pr, vr = to_flux(ur, fr)
        rhol, rhor = ul[0], ur[0]
        el, er = ul[2], ur[2]
        hl, hr = (el+pl) / rhol, (er+pr) / rhor
        al, ar = np.sqrt(gamma*pl/rhol), np.sqrt(gamma*pr/rhor)
        Ml, Mr = vl/al, vr/ar      

        # Split flux(+) component
        if Ml <= -1:
            flp[:] = 0
        elif Ml >= 1:
            flp = fl
        elif -1 < Ml < 0:
            flp = np.array([0.5*rhol/gamma * (vl+al),
                            0.5*rhol/gamma * (vl+al)*(vl+al),
                            0.5*rhol/gamma * (vl+al)*(hl + vl*al)])
        else :
            flp = np.array([0.5*rhol/gamma * (2*(gamma-1)*vl + vl + al),
                            0.5*rhol/gamma * (2*(gamma-1)*vl**2 + (vl+al)**2), 
                            0.5*rhol/gamma * ((gamma-1)*vl**3 + (hl + vl*al)*(vl+al))])
        
        # Split flux(-) component
        if Mr <= -1:
            frm = fr
        elif -1 < Mr < 0:
            frm = np.array([0.5*rhor/gamma * (2*(gamma-1)*vr + vr - ar),
                            0.5*rhor/gamma * (2*(gamma-1)*vr**2 + (vr-ar)**2),
                            0.5*rhor/gamma * ((hr - vr*ar)*(vr-ar) + (gamma-1)*vr**3)])
        elif 0 <= Mr < 1:
            frm = np.array([0.5*rhor/gamma * (vr-ar),
                            0.5*rhor/gamma * (vr-ar)*(vr-ar),
                            0.5*rhor/gamma * (vr-ar)*(hr - vr*ar)]) 
        else:
            frm[:] = 0
        
        # Assemble!    
        for jdx in range(nvars):
            fn[jdx] = flp[jdx] + frm[jdx]
        
    return _flux

# van Leer scheme
def vanleer(gamma=1.4, nvars=3):
    fl, fr = np.empty(nvars), np.empty(nvars)
    
    def _flux(ul, ur, fn):
        flp, frm = np.empty(nvars), np.empty(nvars)
        
        # Variables
        pl, vl = to_flux(ul, fl)
        pr, vr = to_flux(ur, fr)
        rhol, rhor = ul[0], ur[0]
        al, ar = np.sqrt(gamma*pl/rhol), np.sqrt(gamma*pr/rhor)
        Ml, Mr = vl/al, vr/ar      
        
        # Split flux(+) component
        if Ml < -1:
            flp[:] = 0.0
        elif Ml > 1:
            flp = fl
        else:
            flp = np.array([0.25*rhol*al*(1+Ml)**2,
                            0.25*rhol*al*(1+Ml)**2 * (0.5*(gamma-1)*Ml + 1) * (2*al/gamma), 
                            0.25*rhol*al*(1+Ml)**2 * (0.5*(gamma-1)*Ml + 1)**2 * 2*al**2 / (gamma**2 - 1)])  
        
        # Split flux(-) component
        if Mr < -1:
            frm = fr
        elif Mr > 1:
            frm[:] = 0
        else:
            frm = np.array([-0.25*rhor*ar*(1-Mr)**2,
                            -0.25*rhor*ar*(1-Mr)**2 * (0.5*(gamma-1)*Mr - 1) * (2*ar/gamma), 
                            -0.25*rhor*ar*(1-Mr)**2 * (0.5*(gamma-1)*Mr - 1)**2 * 2*ar**2 / (gamma**2 - 1)])
        
        # Assemble!
        for jdx in range(nvars):
            fn[jdx] = flp[jdx] + frm[jdx]  
            
    return _flux              
    
# AUSM+ scheme
# def ausmp(alp=3/16, b=0.125, gamma=1.4, nvars=3):
#     fl, fr = np.empty(nvars), np.empty(nvars)
    
#     def _flux(ul, ur, fn):
#         # Variables
#         pl, vl = to_flux(ul, fl)
#         pr, vr = to_flux(ur, fr)
#         rhol, rhor = ul[0], ur[0]
#         al, ar = np.sqrt(gamma*pl/rhol), np.sqrt(gamma*pr/rhor)
#         el, er = ul[2], ur[2]
#         hl, hr = (el+pl) / rhol, (er+pr) / rhor

#         a = 0.5*(al+ar)
#         rho_l = max(rhol, 1e-12)
#         rho_r = max(rhor, 1e-12)
#         p_l   = max(pl, 1e-12)
#         p_r   = max(pr, 1e-12)

#         al = np.sqrt(gamma * p_l / rho_l)
#         ar = np.sqrt(gamma * p_r / rho_r)

#         a = max(0.5*(al + ar), 1e-12)
#         Ml, Mr = vl/a, vr/a
        
#         if abs(Ml) <= 1:
#             Mp = 0.25*(Ml+1)**2 + b*(Ml**2-1)**2
#             Pp = 0.25*pl*(Ml+1)**2*(2-Ml) + alp*pl*Ml*(Ml**2-1)**2
#         else:
#             Mp = 0.5*(Ml + abs(Ml))
#             Pp = 0.5*pl*(1 + np.sign(Ml))
#             # if Ml > 0:
#             #     Pp = 0.5*pl*(1+Ml)
#             # else:
#             #     Pp = 0.5*pl*(1-Ml)
                
#         if abs(Mr) <= 1:
#             Mm = -0.25*(Mr-1)**2 - b*(Mr**2-1)**2
#             Pm = 0.25*pr*(Mr-1)**2*(2+Mr) - alp*pr*Mr*(Mr**2-1)**2
#         else:
#             Mm = 0.5*(Mr - abs(Mr))
#             Pm = 0.5*pr*(1 - np.sign(Mr))
#             # if Mr > 0:
#             #     Pm = 0.5*pr*(1-Mr)
#             # else:
#             #     Pm = 0.5*pr*(1+Mr)
        
#         M_half = Mp + Mm
#         P_half = Pp + Pm

#         mdot = M_half * a
#         # ===== Upwind selection =====
#         if mdot >= 0:
#             rho = rhol
#             v   = vl
#             h   = hl
#         else:
#             rho = rhor
#             v   = vr
#             h   = hr

#         # ===== Flux =====
#         fn[0] = mdot
#         fn[1] = mdot * v + P_half
#         fn[2] = mdot * h

#     return _flux

def ausm(gamma=1.4, nvars=3):

    fl, fr = np.empty(nvars), np.empty(nvars)
    eps = 1e-12

    def _flux(ul, ur, fn):

        # ===== Variables =====
        pl, vl = to_flux(ul, fl)
        pr, vr = to_flux(ur, fr)

        pl = max(pl, eps)
        pr = max(pr, eps)
        rhol = max(ul[0], eps)
        rhor = max(ur[0], eps)

        el, er = ul[2], ur[2]
        
        # el = max(ul[2], 0.5*rhol*vl**2 + eps)
        # er = max(ur[2], 0.5*rhor*vr**2 + eps)

        hl = (el + pl) / rhol
        hr = (er + pr) / rhor

        # al = np.sqrt(gamma * pl / rhol)
        # ar = np.sqrt(gamma * pr / rhor)
        
        al = np.sqrt(max(gamma*pl/rhol, eps))
        ar = np.sqrt(max(gamma*pr/rhor, eps))

        a_face = max(0.5 * (al + ar), eps)

        ml = vl / a_face
        mr = vr / a_face

        mp = 0.5 * (ml + abs(ml))
        mm = 0.5 * (mr - abs(mr))

        mface = mp + mm

        pp = 0.5 * (1 + np.sign(ml)) * pl
        pm = 0.5 * (1 - np.sign(mr)) * pr

        pface = pp + pm

        mplus = max(mface, 0.0)
        mminus = min(mface, 0.0)

        fn[0] = a_face * (mplus * rhol + mminus * rhor)
        fn[1] = a_face * (mplus * rhol * vl + mminus * rhor * vr) + pface
        fn[2] = a_face * (mplus * rhol * hl + mminus * rhor * hr)

    return _flux

def ausmp(gamma=1.4, nvars=3):

    fl, fr = np.empty(nvars), np.empty(nvars)
    beta = 1.0 / 8.0
    alpha = 3.0 / 16.0
    eps = 1e-12

    def _flux(ul, ur, fn):
            
        # Get variables    
        pl, vl = to_flux(ul, fl)
        pr, vr = to_flux(ur, fr)
        pl, pr = max(pl,eps), max(pr,eps)
        rhol, rhor = max(ul[0],0), max(ur[0],0)
        el, er = ul[2], ur[2]
        hl, hr = (el+pl)/rhol, (er+pr)/rhor

        # sonic speed
        astar2l,astar2r = hl*2*(gamma-1)/(gamma+1), hr*2*(gamma-1)/(gamma+1)
        astarl, astarr = np.sqrt(max(astar2l, eps)), np.sqrt(max(astar2r, eps))

        atil_l = astar2l/max(astarl, abs(vl), eps)
        atil_r = astar2r/max(astarr, abs(vr), eps)

        # a_face = min(atil_l, atil_r)
        a_face = max(atil_l, atil_r)
        a_face = max(a_face, eps)

        # mach numbers
        ml = vl/a_face
        mr = vr/a_face

        # convective blending mach numbers
        if abs(ml) >= 1:
            mp = 0.5*(ml + abs(ml))
        else:
            mp = 0.25*(ml+1)**2 + beta*(ml**2-1)**2

        if abs(mr) >=1:
            mm = 0.5*(mr - abs(mr))
        else:
            mm = -0.25*(mr-1)**2 - beta*(mr**2-1)**2

        mface = mp + mm

        # pressure blending
        if abs(ml) >= 1:
            pp = 0.5*(1+np.sign(ml))
        else:
            pp = 0.25*(ml+1)**2*(2-ml) + alpha*ml*(ml**2-1)**2
        
        if abs(mr) >= 1:
            pm = 0.5*(1-np.sign(mr))
        else:
            pm = 0.25*(mr-1)**2*(2+mr) - alpha*mr*(mr**2-1)**2
        
        pface = pp*pl + pm*pr

        # convective flux
        mplus = max(mface,0.0)
        mminus = min(mface, 0.0)
        fn[0] = a_face * (mplus*rhol + mminus*rhor)
        fn[1] = a_face * (mplus*rhol*vl + mminus*rhor*vr) + pface
        fn[2] = a_face * (mplus*rhol*hl +mminus*rhor*hr)
        
    return _flux

# Flux Difference Splitting    
def roe_basic(gamma=1.4, nvars=3):
    fl, fr = np.empty(nvars), np.empty(nvars)
     # Entropy fix
    def entropy_fix(lam, eps):
        if abs(lam) < eps:
            return (lam**2 + eps**2) / (2*eps)
        else:
            return abs(lam)
    def _flux(ul, ur, fn):
        pl, vl = to_flux(ul, fl)
        pr, vr = to_flux(ur, fr)

        rhol, rhor = ul[0], ur[0]
        El, Er = ul[2], ur[2]

        hl = (El + pl) / rhol
        hr = (Er + pr) / rhor

        srhol = np.sqrt(rhol)
        srhor = np.sqrt(rhor)

        ssrho = srhol + srhor

        u_roe = (srhol * vl + srhor * vr) / ssrho
        h_roe = (srhol * hl + srhor * hr) / ssrho

        a_roe = np.sqrt((gamma - 1.0) * (h_roe - 0.5 * u_roe**2))
        
        eps = 0.2 * a_roe
        
        lam = np.array([u_roe - a_roe, u_roe, u_roe + a_roe])
        lam_fix = np.empty(3)
        lam_fix[0] = entropy_fix(lam[0], eps)
        lam_fix[1] = entropy_fix(lam[1], eps)
        lam_fix[2] = entropy_fix(lam[2], eps)

        k = np.empty((3, 3))
        k[0, :] = [1, 1, 1]
        k[1, :] = [u_roe - a_roe, u_roe, u_roe + a_roe]
        k[2, :] = [h_roe - u_roe*a_roe, 0.5*u_roe**2, h_roe + u_roe*a_roe]

        du1 = rhor - rhol
        du2 = rhor * vr - rhol * vl
        du5 = Er - El

        alp2 = (gamma - 1) * (du1*(h_roe - u_roe**2) + u_roe*du2 - du5) / a_roe**2
        alp1 = (du1*(u_roe + a_roe) - du2 - a_roe*alp2) / (2*a_roe)
        alp3 = du1 - (alp1 + alp2)

        for jdx in range(nvars):
            fn[jdx] = 0.5 * (fl[jdx] + fr[jdx]) \
                    - 0.5 * (alp1 * abs(lam_fix[0]) * k[jdx, 0] +
                             alp2 * abs(lam_fix[1]) * k[jdx, 1] +
                             alp3 * abs(lam_fix[2]) * k[jdx, 2])

    return _flux

# Roe hybrid(+HLLE) scheme
def roe(gamma=1.4, nvars=3):  
    fl, fr = np.empty(nvars), np.empty(nvars)
     
     # Entropy fix
    def entropy_fix(lam, eps):
        if abs(lam) < eps:
            return (lam**2 + eps**2) / (2*eps)
        else:
            return abs(lam)
        
    def _flux(ul, ur, fn):
        # Variables
        pl, vl = to_flux(ul, fl)
        pr, vr = to_flux(ur, fr)
        rhol, rhor = ul[0], ur[0]
        el, er = ul[2], ur[2]

        # positivity check → HLLE
        if (rhol <= 1e-12 or rhor <= 1e-12 or
            pl   <= 1e-12 or pr   <= 1e-12):

            al = np.sqrt(abs(gamma*pl/rhol)) if rhol > 0 else 0.0
            ar = np.sqrt(abs(gamma*pr/rhor)) if rhor > 0 else 0.0

            # HLLE flux
            # Wave speed modication
            SL = min(vl - al, vr - ar)
            SR = max(vl + al, vr + ar)

            if SL >= 0:
                fn[:] = fl
            elif SR <= 0:
                fn[:] = fr
            else:
                for jdx in range(nvars):
                    fn[jdx] = (SR*fl[jdx] - SL*fr[jdx] + SL*SR*(ur[jdx] - ul[jdx])) / (SR - SL)
            return

        al = np.sqrt(gamma*pl/rhol)
        ar = np.sqrt(gamma*pr/rhor)

        hl, hr = (el+pl)/rhol, (er+pr)/rhor

        srhol = np.sqrt(rhol)
        srhor = np.sqrt(rhor)

        ssrho = srhol + srhor
        u_roe = (srhol*vl + srhor*vr) / ssrho
        h_roe = (srhol*hl + srhor*hr) / ssrho

        a2 = (gamma - 1.0)*(h_roe - 0.5*u_roe**2)
        a2 = max(a2, 1e-12)
        a_roe = np.sqrt(a2)

        # Roe eigenvalue fix
        # Einfeldt wave speed
        lam1 = min(vl - al, u_roe - a_roe)
        lam2 = u_roe
        lam3 = max(vr + ar, u_roe + a_roe)
        
        # entropy fix
        eps = 0.2 * a_roe

        lam_fix = np.empty(3)
        lam_fix[0] = entropy_fix(lam1, eps)
        lam_fix[1] = entropy_fix(lam2, eps)
        lam_fix[2] = entropy_fix(lam3, eps)

        k = np.empty((3,3))
        k[0, :] = [1, 1, 1]
        k[1, :] = [u_roe-a_roe, u_roe, u_roe+a_roe]
        k[2, :] = [h_roe-(u_roe*a_roe), 0.5*u_roe**2, h_roe+(u_roe*a_roe)]

        du1 = rhor - rhol
        du2 = rhor * vr - rhol * vl
        du5 = er - el

        alp2 = (gamma-1) * (du1*(h_roe - u_roe**2) + u_roe*du2 - du5) / a_roe**2
        alp1 = (du1 * (u_roe + a_roe) - du2 - a_roe*alp2) / (2*a_roe)
        alp5 = du1 - (alp1 + alp2)

        for jdx in range(nvars):
            fn[jdx] = 0.5 * (fl[jdx] + fr[jdx]) \
                    - 0.5 * (
                        alp1 * lam_fix[0] * k[jdx, 0] +
                        alp2 * lam_fix[1] * k[jdx, 1] +
                        alp5 * lam_fix[2] * k[jdx, 2])    
    return _flux

# HLL scheme
def hll(gamma=1.4, nvars=3):  
    fl, fr = np.empty(nvars), np.empty(nvars)
    def _flux(ul, ur, fn):
       # Variables
        eps = 1e-6
        pl, vl = to_flux(ul, fl)
        pr, vr = to_flux(ur, fr)
        rhol, rhor = ul[0], ur[0]
        
        al, ar = np.sqrt(gamma*pl/rhol), np.sqrt(gamma*pr/rhor)
        el, er = ul[2], ur[2]
        hl, hr = (el+pl) / rhol, (er+pr) / rhor  

        R = np.sqrt(rhol/rhor)
        u_roe = (R*vr + vl)/(R + 1)
        H_roe = (R*hr + hl)/(R + 1)
        a_roe = np.sqrt((gamma - 1)*(H_roe - 0.5*u_roe**2))
        
        SL = u_roe - a_roe
        SR = u_roe + a_roe
        
        for jdx in range(nvars):
            if SL >= 0:
                fhll = fl
            elif SL < 0 < SR:
                fhll = (SR*fl - SL*fr + SL*SR*(ur - ul)) / (SR - SL)
            elif SR <= 0:
                fhll = fr
            fn[jdx] = fhll[jdx]
    return _flux

# HLLC scheme
def hllc(gamma=1.4, nvars=3):
        fl, fr = np.empty(nvars), np.empty(nvars)
        def _flux(ul, ur, fn):
            # Variables
            pl, vl = to_flux(ul, fl)
            pr, vr = to_flux(ur, fr)
            rhol, rhor = ul[0], ur[0]
            el, er = ul[2], ur[2]
            al = np.sqrt(gamma * pl / rhol)
            ar = np.sqrt(gamma * pr / rhor)

            R = np.sqrt(rhol/rhor)
            u_roe = (R * vr + vl) / (R + 1)
            H_roe = (R*(er+pr)/rhor + (el+pl)/rhol) / (R + 1)
            a_roe = np.sqrt((gamma - 1) * (H_roe - 0.5 * u_roe**2))

            SL = min(vl - al, u_roe - a_roe)
            SR = max(vr + ar, u_roe + a_roe)

            S_star = (pr-pl + rhol*vl*(SL-vl) - rhor*vr*(SR-vr)) / (rhol*(SL-vl) - rhor*(SR - vr))

            # Left star
            rhol_star = rhol * (SL-vl) / (SL-S_star)
            el_star = (el/rhol + (S_star-vl)*(S_star + pl/(rhol*(SL-vl)))) * rhol_star

            # Right star
            rhor_star = rhor * (SR-vr) / (SR-S_star)
            er_star = (er/rhor + (S_star-vr)*(S_star + pr/(rhor*(SR-vr)))) * rhor_star

            if SL >= 0:
                fn[:] = fl

            elif SL <= 0 <= S_star:
                fn[0] = fl[0] + SL*(rhol_star - rhol)
                fn[1] = fl[1] + SL*(rhol_star*S_star - rhol*vl)
                fn[2] = fl[2] + SL*(el_star - el)

            elif S_star <= 0 <= SR:
                fn[0] = fr[0] + SR*(rhor_star - rhor)
                fn[1] = fr[1] + SR*(rhor_star*S_star - rhor*vr)
                fn[2] = fr[2] + SR*(er_star - er)

            else:  # SR <= 0
                fn[:] = fr

        return _flux
        