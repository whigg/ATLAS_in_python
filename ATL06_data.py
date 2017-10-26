# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 10:46:21 2017

Class to read and manipulate ATL06 data.  Currently set up for Ben-style fake data, should be modified to work with the official ATL06 prodct foramt

@author: ben
"""
import h5py
import numpy as np

class ATL06_data:
    np.seterr(invalid='ignore')
    def __init__(self, filename=None, x_bounds=None, y_bounds=None, list_of_fields=None, list_of_data=None, from_dict=None):
        if list_of_fields is None:
            # read everything
            list_of_fields=['x_RGT','y_RGT','h_LI','h_LI_sigma','lat_ctr','lon_ctr','time','h_robust_spread','signal_selection_source','SNR_significance']
        self.list_of_fields=list_of_fields
        if list_of_data is not None:
            self.build_from_list_of_data(list_of_data)
            return None     
        if from_dict is not None:
            self.list_of_fields=list_of_fields
            for field in list_of_fields:
                self=setattr(self, from_dict[field])
            return
        # read from a file if specified
        if filename is not None:
            # read a list of files if list provided
            if isinstance(filename, (list, tuple)):
                D6_list=[ATL06_data(filename=thisfile, x_bounds=x_bounds, y_bounds=y_bounds, list_of_fields=list_of_fields) for thisfile in filename]
                self.build_from_list_of_data(D6_list)
            elif isinstance(filename, (basestring)):
                # this happens when the input filename is a string, not a list
                self.read_from_file(filename, x_bounds=x_bounds, y_bounds=y_bounds)
            else:
                raise TypeError
        else:
            # no file specified, set blank
            for field in list_of_fields:
                setattr(self, field, np.zeros((2,0)))       
          
    def read_from_file(self, filename, x_bounds=None, y_bounds=None, list_of_fields=None):
        if list_of_fields is None:
            # read everything
            list_of_fields=self.list_of_fields
        h5_f=h5py.File(filename,'r')
 
        for field in list_of_fields:
            #print field
            try:
                setattr(self, field, np.array(h5_f[field]).transpose())
            except KeyError:
                setattr(self, field, np.zeros_like(self.time)+np.NaN)
        # build the ATL06 quality flag
        quality=np.array(h5_f['signal_selection_source']).transpose()>1.
        quality=np.logical_or(quality, np.array(h5_f['h_robust_spread']).transpose()>1.)
        quality=np.logical_or(quality, np.array(h5_f['h_LI_sigma']).transpose()>1.)
        quality=np.logical_or(quality, np.array(h5_f['SNR_significance']).transpose()>0.02)        
        self.atl06_quality_summary=quality
        if 'atl06_quality_summary' not in self.list_of_fields:
            self.list_of_fields.append('atl06_quality_summary')
        # read the rest of the fields.
        return

    def append(self, D):
        for field in self.list_of_fields:
            setattr(self, np.c_[getattr(self, field), getattr(D, field)])
        return        

    def build_from_list_of_data(self, D6_list):
        try:
            for field in self.list_of_fields:
                data_list=[getattr(this_D6, field) for this_D6 in D6_list]       
                setattr(self, field, np.concatenate(data_list, 0))
        except TypeError:
            for field in self.list_of_fields:
                setattr(self, field, getattr(D6_list, field))
        return 
    
    def index(self, index):
        for field in self.list_of_fields:
            setattr(self, field, getattr(self, field)[index,:])
        return
        
    def subset(self, index):
        dd=dict()
        for field in self.list_of_fields:
            dd[field]=getattr(self, field)[index,:]
        return ATL06_data(from_dict=dd, list_of_fields=self.list_of_fields)
            
    def copy(self):
        return ATL06_data(list_of_data=(self), list_of_fields=self.list_of_fields)
    
    def plot(self):
        import matplotlib.pyplot as plt
        colors=('r','b')
        markers=['o','x']
        flag_vals=(0, 1)
        marker_sizes=(6, 6)
        for col in (0, 1):
            for flag_val in flag_vals:
                these=self.atl06_quality_summary[:,col]==flag_val            
                #print 'found %d/%d points' %(np.sum(these), these.shape[0])
                if np.any(these):
                    plt.errorbar(self.x_RGT[these,col], self.h_LI[these,col], yerr=self.h_LI_sigma[these, col], c=colors[col], marker=markers[flag_val], linestyle='None', markersize=marker_sizes[flag_val]);
        #temp=self.h_LI[1,self.ATL06_quality_summary[:,1]==1,1]
        plt.ylim(np.amin(self.h_LI[self.atl06_quality_summary[:,1]==0,1])-5., np.amax(self.h_LI[self.atl06_quality_summary[:,1]==0 ,1])+5 ) 
        #plt.show()
        return
    def get_pairs(self):
        pair_list=list()
        unpaired_segments=list()
        # loop over segment and orbit numbers         
        for rep, seg in zip(set(self.orbit_number[:,0]), set(self.seg_count[:,0])):
            these=np.where(np.logical_and(self.orbit_number==rep, self.seg_count=seg))
            if len(these)>0:
                pair_list.append(ATL06_pair(D6.index(these)))
        return pair_list

class ATL06_pair:
    def __init__(self, D6):
        #initializes based on input D6, assumed to contain one pair
        self.x_atc=np.mean(D6.x_RGT)
        self.y_atc=np.mean(D6.y_RGT)
        self.dh_dx=D6.dh_dif_dx
        self.dh_dx.shape=[1,2]
        self.dh_dy=np.mean(D6.dh_fit_dy)
        self.time=np.mean(D6.time)
        self.seg_count=np.mean(D6.seg_count)
        self.rep=np.mean(D6.orbit_number)
        self.h=D6.h_LI
        self.h.shape=[1,2]
        ## repeat, 
        # need to define a method to map the segment validity 