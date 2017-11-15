# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 16:24:11 2017

@author: ben
"""
from glob import glob
import numpy as np
from ATL06_to_ATL11 import fit_ATL11
import sys
 
np.seterr(invalid='ignore')

#try:
#    del sys.modules['ATL06_data']; 
#except:
#    pass
#from ATL06_data import ATL06_data 
filenames=glob('/Volumes/ice1/ben/sdt/ATLxx_example/PIG_Collab_v13A/ATL06/run_1/rep_*/Track_462_D3.h5')
#filenames=filenames[0:3] 
x_ctr=33044510.0
# NEED TO TRY WITH THESE FILES AND PAIR=1
fit_ATL11(filenames, pair=3, seg_x_centers=x_ctr+np.arange(0, 60, 120))
