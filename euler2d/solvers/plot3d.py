import numpy as np
import struct


def read_plot3d_b(fname, dtype=np.float64):
    """
    Plot 3D Grid Reader (Single block)
    
    Parameters
    ----------
    fname : string
        격자 이름
    dtype : np.float32 | np.float64
        형식
    """
    with open(fname, 'rb') as f:
        nblocks = struct.unpack("I", f.read(4))[0]
        if nblocks != 1:
            raise ValueError('Only single block can be read')
        
        # Read index
        imax = struct.unpack("I", f.read(4))[0]
        jmax = struct.unpack("I", f.read(4))[0]
        kmax = struct.unpack("I", f.read(4))[0]
        
        # Read grid points
        ndims = 3
        x = np.fromfile(f, dtype)
        
        return x.reshape(ndims, kmax, jmax, imax)
        
        
def write_q(fname, q, mach=1.0, alpha=0.0, reyn=-1, time=0.0):
    """
    Plot 3D solution writer (2D)
    
    Parameter
    ---------
    fname : string
        파일 이름
    q : array
        Conservative variable
    """
    nblocks = 1
    
    if len(q.shape) < 4:
        is_2d = True
    else:
        is_2d = False
    
    # Average    
    if is_2d:
        # Corner point correction
        q[:, 0, 0] = (q[:, 0, 1] + q[:, 1, 0] + q[:, 1, 1])/3
        q[:, 0,-1] = (q[:, 0,-2] + q[:, 1,-1] + q[:, 1,-2])/3
        q[:,-1, 0] = (q[:,-1, 1] + q[:,-2, 0] + q[:,-2, 1])/3
        q[:,-1,-1] = (q[:,-1,-2] + q[:,-2,-1] + q[:,-2,-2])/3
        
        qp = (q[:, :-1, :-1] + q[:, :-1, 1:] + q[:, 1:, :-1] + q[:, 1:, 1:])/4
        kmax = 1
        nvars, jmax, imax = qp.shape
    else:
        qp = (
            q[:, :-1, :-1, :-1] + q[:, :-1, :-1, 1:] + q[:, -1:, 1:, :-1] + q[:,-1:, 1:, 1:] +
            q[:, 1:, :-1, :-1] + q[:, 1:, :-1, 1:] + q[:, 1:, 1:, :-1] + q[:, 1:, 1:, 1:]
        )/8
        nvars, kmax, jmax, imax = qp.shape
        
    with open(fname, 'wb') as f:
        # Write index
        f.write(struct.pack("I", nblocks))
        f.write(struct.pack("I", imax))
        f.write(struct.pack("I", jmax))
        f.write(struct.pack("I", kmax))
        
        f.write(struct.pack("d", mach))
        f.write(struct.pack("d", alpha))
        f.write(struct.pack("d", reyn))
        f.write(struct.pack("d", time))
        
        # Write grid points
        qp[0].tofile(f)
        qp[1].tofile(f)
        qp[2].tofile(f)
        
        if is_2d:
            # Dummpy data for z-momentum
            w = np.zeros_like(qp[0]).tofile(f)
        
        qp[3].tofile(f)
        
        if not is_2d:
            qp[4].tofile(f)
