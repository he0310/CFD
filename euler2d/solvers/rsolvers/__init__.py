from solvers.fluid import to_flux

# Import implemented schemes!!!
from solvers.rsolvers.roe import make_roe
from solvers.rsolvers.rusanov import make_rusanov
from solvers.rsolvers.hll import make_hllc
from solvers.rsolvers.ausm import make_ausmp


import re


def get_rsolver(name, *args, **kwargs):
    """
    Ge Rsolvers
    ------------
    
    Parameters
    ----------
    name : string
        name of scheme
        
    Return
    ------
    flux : jitted function
        kernel from 'make_(name)' generator
    """
    print('Riemann Solver : {}'.format(name))
    
    # replace + in name with 'p' (ex. ausm+ -> ausmp)
    fname = re.sub(r'\+', 'p', name)
    
    # Run the kernal generator make_(name)
    flux = eval('make_' + fname)(*args, **kwargs)
    
    # Return jitted kernel
    return flux
