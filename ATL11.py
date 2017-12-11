# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 11:08:33 2017

@author: ben
"""

import numpy as np
from poly_ref_surf import poly_ref_surf
import matplotlib.pyplot as plt 

class generic_group:
    def __init__(self, N_ref_pts, N_reps, per_pt_fields, full_fields):
        for field in per_pt_fields:
            setattr(self, field, np.zeros([N_ref_pts, 1]))
        for field in full_fields:
            setattr(self, field, np.zeros([N_ref_pts, N_reps]))
        self.per_pt_fields=per_pt_fields
        self.full_fields=full_fields
        self.list_of_fields=self.per_pt_fields.append(self.full_fields)
        

       
class valid_mask:
    def __init__(self, dims, fields):
        for field in fields:
            #print('field=',field)
            setattr(self, field, np.zeros(dims, dtype='bool'))

class ATL11_data:
    def __init__(self, N_ref_pts, N_reps):
        self.Data=[]
        # define empty records here based on ATL11 ATBD
        self.corrected_h=generic_group(N_ref_pts, N_reps, ['ref_pt_lat', 'ref_pt_lon', 'ref_pt_number'], ['mean_pass_time', 'pass_h_shapecorr', 'pass_h_shapecorr_sigma','pass_h_shapecorr_sigma_systematic','quality_summary'])
        
class ATL11_point:
    def __init__(self, N_pairs=1, x_atc_ctr=np.NaN,  y_atc_ctr=np.NaN, track_azimuth=np.NaN, max_poly_degree=[1, 1], N_reps=12):
        self.x_atc_ctr=x_atc_ctr
        self.y_atc_ctr=y_atc_ctr
        self.z_poly_fit=None
        self.mx_poly_fit=None
        self.my_poly_fit=None
        self.valid_segs =valid_mask((N_pairs,2),  ('data','x_slope','y_slope' ))  #  2 cols, boolan, all F to start
        self.valid_pairs=valid_mask((N_pairs,1), ('data','x_slope','y_slope', 'all','ysearch'))  # 1 col, boolean
        self.unselected_cycle_segs=np.zeros((N_pairs,2), dtype='bool')
        self.z=ATL11_data(1, N_reps)
        self.status=dict()

    def select_ATL06_pairs(self, D6, pair_data, x_polyfit_ctr, params_11):
        # this is section 5.1.2: select pairs for reference-surface calculation    
        # step 1a:  Select segs by data
        self.valid_segs.data[np.where(D6.atl06_quality_summary==0)]=True
        # step 1b; the backup step here is UNDOCUMENTED AND UNTESTED
        if not np.any(self.valid_segs.data):
            self.status['atl06_quality_summary_all_nonzero']=1.0
            self.valid_segs.data[np.where(np.logical_or(D6.snr_significance<0.02, D6.signal_selection_source <=2))]=True
            if not np.any(self.valid_segs.data):
                self.status['atl06_quality_all_bad']=1
                return 
                
        seg_sigma_threshold=np.maximum(params_11.seg_sigma_threshold_min, 3*np.median(D6.h_li_sigma[np.where(self.valid_segs.data)]))
        self.status['N_above_data_quality_threshold']=np.sum(D6.h_li_sigma<seg_sigma_threshold)
        self.valid_segs.data=np.logical_and( self.valid_segs.data, D6.h_li_sigma<seg_sigma_threshold)
        self.valid_segs.data=np.logical_and( self.valid_segs.data , np.isfinite(D6.h_li_sigma))    
        
        # step 1c map valid_segs.data to valid_pairs.data
        # ATL06 data fields are Nx2, each rows is a pair
        self.valid_pairs.data=np.logical_and(self.valid_segs.data[:,0], self.valid_segs.data[:,1])
        if not np.any(self.valid_pairs.data):
            self.status['no_valid_pairs']=1
            return 
        
        # 2b. Calculate the y center of the slope regression
        self.y_polyfit_ctr=np.median(pair_data.y[self.valid_pairs.data])
        
        # 2c. identify segments close enough to the y center     
        self.valid_pairs.ysearch=np.abs(pair_data.y.ravel()-self.y_polyfit_ctr)<params_11.L_search_XT  
        
        # 3a: combine data and ysearch
        pairs_valid_for_y_fit=np.logical_and(self.valid_pairs.data.ravel(), self.valid_pairs.ysearch.ravel()) 
        # 3b:choose the degree of the regression for across-track slope
        if len(np.unique(pair_data.x[pairs_valid_for_y_fit]))>1:
            my_regression_x_degree=1
        else:
            my_regression_x_degree=0
        if len(np.unique(pair_data.y[pairs_valid_for_y_fit]))>1:
            my_regression_y_degree=1
        else:
            my_regression_y_degree=0
    
        #3c: Calculate yslope_regression_tol
        y_slope_sigma=np.sqrt(np.sum(D6.h_li_sigma[pairs_valid_for_y_fit,:]**2, axis=1))/np.transpose(np.diff(D6.y_atc[pairs_valid_for_y_fit,:], axis=1)) 
        print('y_slope_sigma ',y_slope_sigma)
        # !!!! for np.maximum two arrays must have same shape
        my_regression_tol=np.maximum(0.01, 3*np.median(y_slope_sigma))
        print('my_regression_tol ',my_regression_tol)
    
        # 3d: regression of yslope against x_pair and y_pair
        self.my_poly_fit=poly_ref_surf(my_regression_x_degree, my_regression_y_degree, x_polyfit_ctr, self.y_polyfit_ctr)
        y_slope_model, y_slope_resid,  y_slope_chi2r, y_slope_valid_flag=self.my_poly_fit.fit(pair_data.x[pairs_valid_for_y_fit], pair_data.y[pairs_valid_for_y_fit], D6.dh_fit_dy[pairs_valid_for_y_fit,0], max_iterations=2, min_sigma=my_regression_tol)
        self.valid_pairs.y_slope[np.where(pairs_valid_for_y_fit),0]=y_slope_valid_flag
        
        #4a. define pairs_valid_for_x_fitparams
        pairs_valid_for_x_fit= np.logical_and(self.valid_pairs.data.ravel(), self.valid_pairs.ysearch.ravel())
       
        # 4b:choose the degree of the regression for along-track slope
        if len(np.unique(D6.x_atc[pairs_valid_for_x_fit,:]))>1:
            mx_regression_x_degree=1
        else:
            mx_regression_x_degree=0
        if len(np.unique(D6.y_atc[pairs_valid_for_x_fit,:].ravel()))>1:
            mx_regression_y_degree=1
        else:
            mx_regression_y_degree=0
    
        #4c: Calculate x-slope_regression_tol
        mx_regression_tol=np.maximum(0.01, 3*np.median(D6.dh_fit_dx_sigma[pairs_valid_for_x_fit,:].flatten())) 
        #print('D6.dh_fit_dx_sigma[pairs_valid_for_x_fit,:] ',D6.dh_fit_dx_sigma[pairs_valid_for_x_fit,:])
        print('D6.dh_fit_dx_sigma ',D6.dh_fit_dx_sigma)
        print('mx_reg_tol is ',mx_regression_tol )
        # 4d-4g: regression of xslope against x_pair and y_pair
        self.mx_poly_fit=poly_ref_surf(mx_regression_x_degree, mx_regression_y_degree, x_polyfit_ctr, self.y_polyfit_ctr) 
        #print(self.my_poly_fit.)
        x_slope_model, x_slope_resid,  x_slope_chi2r, x_slope_valid_flag=self.mx_poly_fit.fit(D6.x_atc[pairs_valid_for_x_fit,:].ravel(), D6.y_atc[pairs_valid_for_x_fit,:].ravel(), D6.dh_fit_dx[pairs_valid_for_x_fit,:].ravel(), max_iterations=2, min_sigma=mx_regression_tol)
        
        # 4e. x_slope_threshold
        #x_slope_threshold = np.max(mx_regression_tol)
        #if np.all(pairs_valid_for_x_fit==x_slope_valid_flag):
        x_slope_valid_flag.shape=[np.sum(pairs_valid_for_x_fit),2]
        self.valid_segs.x_slope[np.where(pairs_valid_for_x_fit),:]=x_slope_valid_flag
        self.valid_pairs.x_slope=np.all(self.valid_segs.x_slope, axis=1) 
        
        # 5: define selected pairss
        self.valid_pairs.all=np.logical_and(self.valid_pairs.data.ravel(), np.logical_and(self.valid_pairs.x_slope.ravel(), self.valid_pairs.y_slope.ravel()))
        #5a: identify unselected cycles
        cycle_selected=D6.cycle[self.valid_pairs.all,:]
        all_cycles=np.unique(D6.cycle.ravel())
        selected_cycles=np.unique(cycle_selected)
        unselected_cycles=np.setdiff1d(all_cycles, selected_cycles)  # make this 2d
        self.unselected_cycle_segs=np.in1d(D6.cycle, unselected_cycles).reshape(D6.cycle.shape)  # (x,2)
        self.unselected_cycle_segs=np.logical_and(self.unselected_cycle_segs, np.logical_and(self.valid_segs.x_slope,self.valid_segs.data))
        return
    def select_y_center(self, D6, pair_data, params):
        y_selected=D6.y_atc[self.valid_pairs.all,:]
        y0=(np.min(y_selected.ravel())+np.max(y_selected.ravel()))/2
        # loop over a range of y centers, select the center with the best score
        y0_shifts=np.round(y0)+np.arange(-100,100, 2)
        score=np.zeros_like(y0_shifts)

        for count, y0_shift in enumerate(y0_shifts):
            #valid_pair_count = np.sum(np.all(np.abs(y_selected-y0_shift)<params.L_search_XT, axis=1)) # np.abs() is bool, np.all checks each row is all True, np.sum sums the Trues, check that we're counting cycles and not pairs
            sel_segs=D6.cycle[np.all(np.abs(D6.y_atc[self.valid_pairs.all,:]-y0_shift)<params.L_search_XT, axis=1)]
            valid_cycle_count=len(np.unique(sel_segs))
            unsel_segs=np.logical_and(np.abs(D6.y_atc-y0_shift)<params.L_search_XT, self.unselected_cycle_segs) #           
            unselected_seg_cycle_count=len(np.unique(D6.cycle[unsel_segs]))
            # the score is equal to the number of cycles with at least one valid pair entirely in the window, 
            # plus 1/100 of the number cycles that contain no valid pairs but have at least one valid segment in the window
            score[count]=valid_cycle_count + unselected_seg_cycle_count/100.
        # identify the y0 vals that correspond the best score
        best = np.argwhere(score == np.amax(score))
        # y0 is the median of these y0 vals                     
        return np.median(y0_shifts[best])    




   
#    for y0_shift, count in enumerate(y0_shifts):
#        score[count]=np.sum(np.abs(y_rep-y0_shift)<params_11.L_search_XT-params_11.beam_spacing/2)+len(set(unselected_segs[np.abs(unselected_segs.y_ATC-y0_shift)<params_11.L_search_XT].rep))/100
    
    
    
         
class ATL11_defaults:
    def __init__(self):
        # provide option to read keyword=val pairs from the input file
        self.L_search_AT=125 # meters, along track (in x), filters along track
        self.L_search_XT=110 # meters, cross track (in y), filters across track
        self.min_slope_tol=0.02 # in degrees?
        self.min_h_tol=0.1  # units? of height offset
        self.seg_sigma_threshold_min=0.05
        #self.y_search=110 # meters
        self.beam_spacing=90 # meters
        self.seg_atc_spacing=20 # meters, segments are 40m long, overlap is 50%