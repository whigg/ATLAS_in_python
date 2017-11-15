# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:52:13 2017

@author: ben
"""
import sys
import numpy as np

np.seterr(invalid='ignore')

try:
    del sys.modules['ATL06_data']; 
except:
    pass
from ATL06_data import ATL06_data; 

filenames=('/home/ben/Dropbox/PIG_ATLAS_v13/ATL06/run_1/rep_1/Track_462-Pair_3_D3.h5', '/home/ben/Dropbox/PIG_ATLAS_v13/ATL06/run_1/rep_2/Track_462-Pair_3_D3.h5')


D=ATL06_data(filename=filenames); 
D.plot()
