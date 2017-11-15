# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 11:08:33 2017

@author: ben
"""

import numpy as np
from poly_ref_surf import poly_ref_surf

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
            setattr(self, field, np.zeros(dims, dtype='bool'))

class ATL11_data:
    def __init__(self, N_ref_pts, N_reps, N_pairs):
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
        self.valid_segs =valid_mask((N_pairs,2),  ('data','x_slope','y_slope' ))
        self.valid_pairs=valid_mask((N_pairs,1), ('data','x_slope','y_slope', 'all'))
        self.unselected_cycle_segs=np.zeros((N_pairs,2), dtype='bool')
        self.D11=ATL11_data(1, N_reps)
        self.status=dict()

    def select_ATL06_pairs(self, D6, pair_data, x_ATC_center, params_11):
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
        self.y_seg_ctr=np.median(pair_data.y[self.valid_pairs.data])
        
        # 2c. identify segments close enough to the y center
        self.valid_pairs.ysearch[:]=np.abs(pair_data.y.ravel()-self.y_seg_ctr)<params_11.L_search_XT
        
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
        sigma_y_slope=np.sqrt(np.sum(D6.h_li_sigma[pairs_valid_for_y_fit,:]**2, axis=1))/np.diff(D6.y_atc[pairs_valid_for_y_fit,:], axis=1)
        my_regression_tol=np.maximum(0.01, 3*np.median(sigma_y_slope))
    
        # 3d: regression of yslope against x_pair and y_pair
        self.my_poly_fit=poly_ref_surf(my_regression_x_degree, my_regression_y_degree, x_ATC_center, self.y_seg_ctr)
        slope_model_y, slope_y_r,  slope_x2r_y, slope_valid_y_flag=self.my_poly_fit.fit(pair_data.x[pairs_valid_for_y_fit], pair_data.y[pairs_valid_for_y_fit], D6.dh_fit_dy[pairs_valid_for_y_fit,0], max_iterations=2, min_sigma=my_regression_tol)
        self.valid_pairs.yslope[np.where(pairs_valid_for_y_fit)]=slope_valid_y_flag
        
        #4a. define pairs_valid_for_x_fit
        pairs_valid_for_x_fit= np.logical_and(self.valid_pairs.data, self.valid_pairs.ysearch)
        
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
    
        # 4d-4g: regression of xslope against x_pair and y_pair
        self.mx_poly_fit=poly_ref_surf(mx_regression_x_degree, mx_regression_y_degree, x_ATC_center, self.y_seg_ctr) 
        slope_model_x, slope_x_r,  slope_x2r_x, slope_valid_x_flag=self.mx_poly_fit.fit(D6.x_atc[pairs_valid_for_x_fit,:].ravel(), D6.y_atc[pairs_valid_for_x_fit,:].ravel(), D6.dh_fit_dx[pairs_valid_for_x_fit,:].ravel(), max_iterations=2, min_sigma=mx_regression_tol)
         
        slope_valid_x_flag.shape=[np.sum(pairs_valid_for_x_fit),2]
        self.valid_segs.xslope[np.where(pairs_valid_for_x_fit),:]=slope_valid_x_flag
        self.valid_pairs.xslope=np.all(self.valid_segs.xslope, axis=1) 
        
        # 5: define selected pairs
        self.valid_pairs.all=np.logical_and(self.valid_pairs.data, np.logical_and(self.valid_pairs.xslope, self.valid_pairs.yslope))
        
        #5a: identify unselected cycles
        cycle_selected=D6.cycle[self.valid_pairs.all,:]
        mask=np.zeros_like(self.h_li, dtype='bool')
        mask[self.valid_segs]=True
        mask[self.valid_pairs.all,:]=False
        all_cycles=np.unique(D6.cycle.ravel())
        selected_cycles=np.unique(cycle_selected)
        unselected_cycles=setdiff1d(all_cycles, selected_cycles)
        self.unselected_cycle_segs=np.in1d(D6.cycle, unselected_cycles)
        return
    def select_y_center(self, D6, pair_data, params):
        y_selected=D6.y_atc[self.valid_pairs.all,:]
        # find the middle of the range of the selected beams
        y0=(np.min(y_selected.ravel())+np.max(y_selected.ravel()))/2
        # loop over a range of y centers, select the center with the best score
        y0_ctrs=np.round(y0)+np.arange(-100,100, 2)
        score=np.zeros_like(y0_ctrs)
        for y0_ctr, count in enumerate(y0_ctrs):
            valid_pair_count=np.sum(np.all(np.abs(y_selected-y0_ctr)<params.y_search, axis=1))
            unsel_segs=np.logical_and(np.abs(D6.y_atc-y0_ctr)<params.y_search, self.unselected_cycle_seg)           
            unselected_seg_cycle_count=len(np.unique(D6.cycle[unsel_segs]))
            # the score is equal to the number of cycles with at least one valid pair entirely in the window, 
            # plus 1/100 of the number cycles that contain no valid pairs but have at least one valid segment in the window
            score[count]=valid_pair_count+unselected_seg_cycle_count/100.
        # identify the y0 vals that correspond the best score
        best=score==np.maximum(score)
        # y0 is the median of these y0 vals                     
        return np.median(y0_ctrs[best])    




   
     for y0_ctr, count in enumerate(y0_ctrs):
        score[count]=np.sum(np.abs(y_rep-y0_ctr)<params.y_search-params.beam_spacing/2)+len(set(unselected_segs[np.abs(unselected_segs.y_ATC-y0_ctr)<params.y_search].rep))/100
    
    
    
         
class ATL11_defaults:
    def __init__(self):
        # provide option to read keyword=val pairs from the input file
        self.L_search_AT=125
        self.L_search_XT=110
        self.min_slope_tol=0.02
        self.min_h_tol=0.1
        self.seg_sigma_threshold_min=0.05