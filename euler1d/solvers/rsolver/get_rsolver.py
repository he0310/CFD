from solvers.fluid import to_flux

# Import implemented schemes!!!
from solvers.rsolvers import rusanov
from solvers.rsolvers import roe 
from solvers.rsolvers import roe_basic
from solvers.rsolvers import stegarwarming
from solvers.rsolvers import vanleer
from solvers.rsolvers import ausmp
from solvers.rsolvers import hll
from solvers.rsolvers import hllc
# Do more

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
    # flux = eval('make_' + fname)(*args, **kwargs)
    flux = eval(fname)(*args, **kwargs)
    
    # Return jitted kernel
    return flux
    
