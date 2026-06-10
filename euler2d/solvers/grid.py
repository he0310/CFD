import numpy as np

def cell_jacobian(x, y):
    """
    Compute Area of cell using inverse Jacobian
    
    Parameters
    ----------
    x : array
        X coordinate (j, i)
    y : array
        Y coordinate (j, i)
        
    Return
    -------
    jac : array
        Jacobian of cell
    """    
    # Parse dimension
    jmax, imax = x.shape
    imax -= 1
    jmax -= 1
    
    # Allocate array
    jac = np.empty((jmax, imax))
    
    # Compute Jacobian
    for j in range(jmax):
        jp = j + 1
        for i in range(imax):
            ip = i +1 
            xch = ((x[jp, ip] + x[j, ip]) - (x[jp, i] + x[j, i]))/2
            ych = ((y[jp, ip] + y[j, ip]) - (y[jp, i] + y[j, i]))/2
            xet = ((x[jp, ip] + x[jp, i]) - (x[j, ip] + x[j, i]))/2
            yet = ((y[jp, ip] + y[jp, i]) - (y[j, ip] + y[j, i]))/2
            
            jac[j, i] = 1/(xch*yet - ych*xet)
            
    return jac


def face_metric(x, y):
    """
    Compute metric of Face (nabla eta /J)
    
    Parameters
    ----------
    x : array
        X coordinate (j, i)
    y : array
        Y coordinate (j, i)
        
    Return
    -------
    si : array
        Face metric along xi direction
    sj : array
        Face metric along eta direction        
    """
    # Parse dimension of coordinate
    jmax, imax = x.shape
    
    # Allocate arrays for \nabla \xi / J, \nabla \eta / J
    si = np.empty((2, jmax-1, imax))
    sj = np.empty((2, jmax, imax-1))
    
    for j in range(jmax-1):
        jp = j + 1
        for i in range(imax):
            si[0, j, i] =  y[jp, i] - y[j, i]
            si[1, j, i] =-(x[jp, i] - x[j, i])
            
    for j in range(jmax):
        for i in range(imax-1):
            ip = i + 1
            sj[0, j, i] =-(y[j, ip] - y[j, i])
            sj[1, j ,i] =  x[j, ip] - x[j, i]
    
    return si, sj


def cell_x(x, y):
    """
    Compute cell center
    
    Parameters
    ----------
    x : array
        X coordinate (j, i)
    y : array
        Y coordinate (j, i)
        
    Return
    -------
    xc : array
        Cell center of X-dir
    yc : array
        Cell center of Y-dir        
    """
    xc = (x[:-1,:-1] + x[1:,:-1] + x[:-1,1:] + x[1:,1:])/4
    yc = (y[:-1,:-1] + y[1:,:-1] + y[:-1,1:] + y[1:,1:])/4
    
    return xc, yc
