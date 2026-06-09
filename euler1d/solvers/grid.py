import numpy as np


def make_uniform_grid1d(nfx, bounds=(0,1)):
    """
    1D uniform grid generator
    
    Parameters
    ----------
    nfx : integer
        격자 (Number of Cell) 개수
    bounds : tuple
        시작점, 끝점
    
    Results
    -------
    xc : ndarray (float)
        격자 중심점 좌표
    dx : float
        격자 크기
    """
    x = np.linspace(*bounds, nfx+1)
    xc = 0.5*(x[:-1] + x[1:])
    dx = x[1:] - x[:-1]
    
    return xc, dx.max()
