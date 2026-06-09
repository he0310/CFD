from numba import types
from numba.extending import intrinsic,register_jitable
from numba.core import cgutils
from numba.typed import Dict

import numpy as np
import numba as nb


@intrinsic
def stack_empty_impl(typingctx,size,dtype):
    def impl(context, builder, signature, args):
        ty=context.get_value_type(dtype.dtype)
        ptr = cgutils.alloca_once(builder, ty,size=args[0])
        return ptr

    sig = types.CPointer(dtype.dtype)(types.int64,dtype)
    return sig, impl


@register_jitable
def stack_empty(size, shape, dtype):
    arr_ptr = stack_empty_impl(size, dtype)
    arr = nb.carray(arr_ptr, shape)
    return arr


@register_jitable
def array(shape, dtype=np.float64):
    # Compute size of shape
    size = 1
    for i in range(len(shape)):
        size *= shape[i]

    arr = stack_empty(size, shape, dtype=dtype)
    return arr