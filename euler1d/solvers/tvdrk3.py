import numpy as np


# def make_tvdrk3(nfx, rhside, nvars=3, npad=1):
#     """
#     TVD Runge Kutta 3rd 계산 커널 생성
    
#     Parameters
#     ----------
#     nfx : integer
#         격자 개수
#     rhside : kernel
#         우변 계산 커널
#     nvars : integer
#         벡터 크기 (기본값: 3)
#     npad : integer
#         Solution 벡터의 Padding 개수
#     """
#     # Temporal arrays
#     f = np.zeros((nvars, nfx+1))
#     du = np.zeros((nvars, nfx+2*npad))
#     u0 = np.zeros((nvars, nfx+2*npad))
    
#     def _step(dt, u):
#         # Implement TVD RK3!!!
#         u0[:] = u

#         rhside(u0, du, f)
#         u1[:] = u0 + dt*du

#         rhside(u1, du, f)
#         u1[:] = 3/4*u0 + 1/4*u1 + 1/4*dt*du

#         rhside(u1, du, f)
#         u[:] = 1/3*u0 + 2/3*u1 + 2/3*dt*du
        
#     return _step

def make_tvdrk3(nfx, rhside, nvars, npad):
    u0 = np.empty((nvars, nfx + 2*npad))
    u1 = np.empty((nvars, nfx + 2*npad))
    u2 = np.empty((nvars, nfx + 2*npad))
    du = np.empty((nvars, nfx + 2*npad))
    f  = np.empty((nvars, nfx + npad))

    def _step(dt, u):
        # Stage 1
        u0[:] = u
        rhside(u0, du, f)
        u1[:] = u0 + dt * du

        # Stage 2
        rhside(u1, du, f)
        u2[:] = 0.75*u0 + 0.25*u1 + 0.25*dt*du

        # Stage 3
        rhside(u2, du, f)
        u[:] = (1.0/3.0)*u0 + (2.0/3.0)*u2 + (2.0/3.0)*dt*du

    return _step
