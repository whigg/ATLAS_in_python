# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 10:46:21 2017

@author: ben
"""
import h5py
import numpy as np

class ATL06_data:
    np.seterr(invalid='ignore')
    def __init__(self, filename=None, x_bounds=None, y_bounds=None, list_of_fields=None ):
        if list_of_fields is None:
            # read everything
            list_of_fields={'x_RGT','y_RGT','h_LI','h_LI_sigma','lat_ctr','lon_ctr','time','h_robust_spread','signal_selection_source','SNR_significance'}
        # read from a file if specified
        if filename is not None:
            # read from a file if specified
            self.read_from_file(filename, x_bounds, y_bounds, list_of_fields)
        else:
            # set blank
            for field in list_of_fields:
                setattr(self, field, np.zeros((2,0)))
        
    def read_from_file(self, filename, x_bounds=None, y_bounds=None, list_of_fields=None):
        if list_of_fields is None:
            # read everything
            list_of_fields={'x_RGT','y_RGT','h_LI','h_LI_sigma','lat_ctr','lon_ctr','time','h_robust_spread','signal_selection_source','SNR_significance','ATL06_quality_summary'}
        h5_f=h5py.File(filename,'r')
        # build the ATL06 quality flag
        quality=np.array(h5_f['signal_selection_source']).transpose()>1.
        quality=np.logical_or(quality, np.array(h5_f['h_robust_spread']).transpose()>1.)
        quality=np.logical_or(quality, np.array(h5_f['h_LI_sigma']).transpose()>1.)
        quality=np.logical_or(quality, np.array(h5_f['SNR_significance']).transpose()>0.02)        
        setattr(self, 'ATL06_quality_summary', quality)
        # read the rest of the fields.
        for field in list_of_fields:
            #print field
            setattr(self, field, np.array(h5_f[field]).transpose())
        return
    def plot(self):
        import matplotlib.pyplot as plt
        colors=('r','b')
        markers=['o','x']
        flag_vals=(0, 1)
        marker_sizes=(6, 6)
        for col in (0, 1):
            for flag_val in flag_vals:
                these=self.ATL06_quality_summary[:,col]==flag_val            
                #print 'found %d/%d points' %(np.sum(these), these.shape[0])
                if np.any(these):
                    plt.errorbar(self.x_RGT[these,col], self.h_LI[these,col], yerr=self.h_LI_sigma[these, col], c=colors[col], marker=markers[flag_val], linestyle='None', markersize=marker_sizes[flag_val]);
        #temp=self.h_LI[1,self.ATL06_quality_summary[:,1]==1,1]
        plt.ylim(np.amin(self.h_LI[self.ATL06_quality_summary[:,1]==0,1])-5., np.amax(self.h_LI[self.ATL06_quality_summary[:,1]==0 ,1])+5 ) 
        #plt.show()
        return