import numba as nb


# Very small number
eps = 1e-10


def get_limiter(method):
    """
    기울기 제한자 함수 생성
    
    Parameter
    ---------
    method : string
        이름
        
    Return
    ------
    limiter : jitted function
        Limiter 함수
    """
    if method == 'minmod':
        print('Minmod limiter')
        f = lambda r: max(0, min(r, 1.0))
    elif method == 'superbee':
        print('Superbee limiter')
        f = lambda r: max(0, min(2.0*r, 1), min(r, 2))
    elif method == 'vanleer':
        print('van Leer limiter')
        def f(r):
            if r <= 0.0:
                return 0.0
            else:
                return 2.0*r / (1.0 + r)
    elif method == 'unlimited':
        print('Unlimited second-order')
        f = lambda r: 1.0
    else:
        print("First-order")
        f = lambda r : 0.0
        
    f_jitted = nb.jit(f, nopython=True)
    
    @nb.jit(nopython=True)
    def limiter(dp, dm):
        if abs(dm) < eps:
            return 1.0
        else:
            return f_jitted(dp/dm)
    
    return limiter
