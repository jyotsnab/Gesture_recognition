# -*- coding: utf-8 -*-
"""
Feature_Engineering.py
@author: jyotsna
"""

"""
Feature Engineering and Feature Extraction

"""

import numpy as np

def stats(ar):
    
    """
    Parameters
    ----------
    ar : 50 * 8 array
    
    Returns
    -------
    Std Dev
    variance
    Mean
    Peak RMS amplitude
    sum of amplitude  = area under curve/window_width 
                        (since window_width is a constant)
    

    """
    return [np.std(ar),np.var(ar), np.mean(ar), np.max(ar),np.sum(ar)]


def hist(ar):
    """
    Parameters
    ----------
    ar : 50 * 8 array
    
    Returns
    -------
    array: 1d
        top 4 amplitude values in the histogram
    """
    a, d= np.histogram(ar,bins=10, density=True)    
    return d[np.argsort(a)[-4:]]

def normalize(ar):
    
    """
    Parameters
    ----------
    ar : 50 * 8 array
    
    Returns
    -------
    array: 50*8 shape
        normalized version of the original sample
    """
    maxval = ar.max()
    minval = ar.min() 
    for i in range(ar.shape[1]):
        ar[:,i] =(ar[:,i]-minval)/(maxval-minval)
    return ar
    