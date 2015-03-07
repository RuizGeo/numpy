
import numpy as np
from numpy.lib import stride_tricks
import time


def rolling2d(a,win_h,win_w,step_h,step_w):

    h,w = a.shape
    shape = ( a.shape[0]*a.shape[1] , win_h , win_w)
    print ((h-win_h)/step_h + 1)  * ((w-win_w)/step_w + 1) , win_h , win_w

    strides = (step_w*a.itemsize, h*a.itemsize,a.itemsize)


    a= np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    print 'a[15,:,:] \n',a[15,:,:]
    return np.einsum('...ijk->...i',a)/(a.shape[-1]*a.shape[-2])
ini=time.time()  
a = np.arange(1000000).reshape(1000,1000)

d= rolling2d(a,3,3,1,1)    
print 'd.shape ', d.shape
d.reshape(1000,1000)
fim = time.time()
print fim-ini, ' tricks' 
