# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 16:31:30 2017

@author: ben
"""


def RDE(x):
    xs=x.copy()
    xs=xs(np.isfinite(xs))
    if len(xs)<2 :
        return np.nan
    ind=np.arange(0.5, len(xs))
    LH=np.interp(np.array([0.16, 0.84])*len(xs), ind, xs.sorted())
    return (LH[1]-LH[0])/2.


#import scipy.stats as stats
#def RDE(x):
#    return (stats.scoreatpercentile(x, 84 )-stats.scoreatpercentile(x, 16))/2.