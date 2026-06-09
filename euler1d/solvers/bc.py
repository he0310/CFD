
def make_bc_zero(nfx, npad=1):
    """
    외삽 경계조건 함수 생성
    
    Parameters
    -----------
    nfx : integer
        격자 개수
    npad : integer
        경계를 위한 padding 개수 (기본값: 1)
        
    Return
    ------
    _run : function
        경계조건 함수
    """
    def _run(u):
        # Non-reflective Boundary Condition
        u[:, npad-1]   = u[:, npad]
        u[:, nfx+npad] = u[:, nfx+npad-1]
        
    return _run
