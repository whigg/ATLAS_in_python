# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 11:08:33 2017

@author: ben
"""

import numpy as np
from poly_ref_surf import poly_ref_surf
import matplotlib.pyplot as plt 
from RDE import RDE
import time

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
        # step 1a:  Select segs by data quality
        self.valid_segs.data[np.where(D6.atl06_quality_summary==0)]=True
        # step 1b; the backup step here is UNDOCUMENTED AND UNTESTED
        if not np.any(self.valid_segs.data):
            self.status['atl06_quality_summary_all_nonzero']=1.0
            self.valid_segs.data[np.where(np.logical_or(D6.snr_significance<0.02, D6.signal_selection_source <=2))]=True
            if not np.any(self.valid_segs.data):
                self.status['atl06_quality_all_bad']=1
                return 
        # 1b: Select segs by height error        
        seg_sigma_threshold=np.maximum(params_11.seg_sigma_threshold_min, 3*np.median(D6.h_li_sigma[np.where(self.valid_segs.data)]))
        self.status['N_above_data_quality_threshold']=np.sum(D6.h_li_sigma<seg_sigma_threshold)
        self.valid_segs.data=np.logical_and( self.valid_segs.data, D6.h_li_sigma<seg_sigma_threshold)
        self.valid_segs.data=np.logical_and( self.valid_segs.data , np.isfinite(D6.h_li_sigma))    
        
        # 1c: Map valid_segs.data to valid_pairs.data
        # ATL06 data fields are Nx2, each rows is a pair
        self.valid_pairs.data=np.logical_and(self.valid_segs.data[:,0], self.valid_segs.data[:,1])
        if not np.any(self.valid_pairs.data):
            self.status['no_valid_pairs']=1
            return 
        
        # 2b. Calculate the y center of the slope regression
        self.y_polyfit_ctr=np.median(pair_data.y[self.valid_pairs.data])
        print('y_polyfit_ctr =',self.y_polyfit_ctr)
        
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
    
        # 3c: Calculate across-track slope regression tolerance
        y_slope_sigma=np.sqrt(np.sum(D6.h_li_sigma[pairs_valid_for_y_fit,:]**2, axis=1))/np.transpose(np.diff(D6.y_atc[pairs_valid_for_y_fit,:], axis=1)).ravel() #same shape as y_slope*
        my_regression_tol=np.max(0.01, 3*np.median(y_slope_sigma))

        for item in range(2):
            # 3d: regression of across-track slope against x_pair and y_pair
            self.my_poly_fit=poly_ref_surf(my_regression_x_degree, my_regression_y_degree, x_polyfit_ctr, self.y_polyfit_ctr)
            y_slope_model, y_slope_resid,  y_slope_chi2r, y_slope_valid_flag=self.my_poly_fit.fit(pair_data.x[pairs_valid_for_y_fit], pair_data.y[pairs_valid_for_y_fit], D6.dh_fit_dy[pairs_valid_for_y_fit,0], max_iterations=2, min_sigma=my_regression_tol)
            plt.figure()
            plt.plot(y_slope_resid,'r.-');plt.ylim([-0.01,0.01])
            titlestr='y_slope_resid for iteration %d' % (item)
            plt.title(titlestr)
            # update what is valid based on regression flag
            self.valid_pairs.y_slope[np.where(pairs_valid_for_y_fit),0]=y_slope_valid_flag                #re-establish pairs_valid for y fit
            # re-establish pairs_valid_for_y_fit
            pairs_valid_for_y_fit=np.logical_and(self.valid_pairs.data.ravel(), self.valid_pairs.y_slope.ravel()) 

            # 3e: calculate across-track slope threshold
            y_slope_threshold=np.max(my_regression_tol,3*RDE(y_slope_sigma))
            
            # 3f: select for across-track residuals within threshold
            self.valid_pairs.y_slope[np.where(pairs_valid_for_y_fit),0]=np.abs(y_slope_resid)<=y_slope_threshold
            # re-establish pairs_valid_for_y_fit
            pairs_valid_for_y_fit=np.logical_and(self.valid_pairs.data.ravel(), self.valid_pairs.y_slope.ravel()) 
                            
        #4a. define pairs_valid_for_x_fit
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
    
        #4c: Calculate along-track slope regression tolerance
        mx_regression_tol=np.maximum(0.01, 3*np.median(D6.dh_fit_dx_sigma[pairs_valid_for_x_fit,:].flatten())) 

        for item in range(2):
            # 4d: regression of along-track slope against x_pair and y_pair
            self.mx_poly_fit=poly_ref_surf(mx_regression_x_degree, mx_regression_y_degree, x_polyfit_ctr, self.y_polyfit_ctr) 
            x_slope_model, x_slope_resid,  x_slope_chi2r, x_slope_valid_flag=self.mx_poly_fit.fit(D6.x_atc[pairs_valid_for_x_fit,:].ravel(), D6.y_atc[pairs_valid_for_x_fit,:].ravel(), D6.dh_fit_dx[pairs_valid_for_x_fit,:].ravel(), max_iterations=2, min_sigma=mx_regression_tol)
            plt.figure()
            plt.plot(x_slope_resid,'.-');plt.ylim([-0.03,0.04]);
            titlestr='x_slope_resid for iteration %d' % (item)
            plt.title(titlestr)
            # update what is valid based on regression flag
            x_slope_valid_flag.shape=[np.sum(pairs_valid_for_x_fit),2]
            self.valid_segs.x_slope[np.where(pairs_valid_for_x_fit),:]=x_slope_valid_flag
            self.valid_pairs.x_slope=np.all(self.valid_segs.x_slope, axis=1) 
            # re-establish pairs_valid_for_x_fit
            pairs_valid_for_x_fit=np.logical_and(self.valid_pairs.data.ravel(), self.valid_pairs.x_slope.ravel()) 
            
            # 4e: calculate along-track slope threshold
            x_slope_threshold = np.max(mx_regression_tol,3*RDE(x_slope_resid))

            # 4f: select for along-track residuals within threshold
            x_slope_resid.shape=[np.sum(pairs_valid_for_x_fit),2]
            self.valid_segs.x_slope[np.where(pairs_valid_for_x_fit),:]=np.transpose(np.tile(np.all(np.abs(x_slope_resid)<=x_slope_threshold,axis=1),(2,1)))
            self.valid_pairs.x_slope=np.all(self.valid_segs.x_slope, axis=1) 
            # re-establish pairs_valid_for_x_fit
            pairs_valid_for_x_fit=np.logical_and(self.valid_pairs.data.ravel(), self.valid_pairs.x_slope.ravel()) 
            
        # 5: define selected pairs
        self.valid_pairs.all=np.logical_and(self.valid_pairs.data.ravel(), np.logical_and(self.valid_pairs.y_slope.ravel(), self.valid_pairs.x_slope.ravel()))
        
        #5a: identify unselected cycles
        cycle_selected=D6.cycle[self.valid_pairs.all,:]
        all_cycles=np.unique(D6.cycle.ravel())
        selected_cycles=np.unique(cycle_selected)
        unselected_cycles=np.setdiff1d(all_cycles, selected_cycles)  # make this 2d
        self.unselected_cycle_segs=np.in1d(D6.cycle, unselected_cycles).reshape(D6.cycle.shape)  # (x,2)
        return
        
    def select_y_center(self, D6, pair_data, params):  #5.1.3
        y_selected=D6.y_atc[self.valid_pairs.all,:]
        cycle_selected=D6.cycle[self.valid_pairs.all,:]
        # find the middle of the range of the selected beams
        y0=(np.min(y_selected.ravel())+np.max(y_selected.ravel()))/2
        # 1: define a range of y centers, select the center with the best score
        y0_shifts=np.round(y0)+np.arange(-100,100, 2)
        score=np.zeros_like(y0_shifts)

        # 2: search for optimal shift val.ue
        for count, y0_shift in enumerate(y0_shifts):
            #print('count ',count, y0_shift)
            sel_segs=np.all(np.abs(y_selected-y0_shift)<params.L_search_XT, axis=1)
            selected_seg_cycle_count=len(np.unique(cycle_selected[sel_segs,:]))
            #print('selected_seg_cycle_count ',selected_seg_cycle_count)
            
            #plt.figure()
            #plt.plot(D6.y_atc[sel_segs,0])
            #plt.plot(y0_shift*np.ones_like(sel_segs),'r')
            #print('sel_segs ',sel_segs)
            #selected_seg_cycle_count=len(np.unique(D6.cycle[sel_segs]))
            
            unsel_segs=np.logical_and(np.abs(D6.y_atc-y0_shift)<params.L_search_XT, self.unselected_cycle_segs) #           
            unselected_seg_cycle_count=len(np.unique(D6.cycle[unsel_segs]))
            
            # the score is equal to the number of cycles with at least one valid pair entirely in the window, 
            # plus 1/100 of the number cycles that contain no valid pairs but have at least one valid segment in the window
            score[count]=selected_seg_cycle_count + unselected_seg_cycle_count/100.
        # 3: identify the y0_shift value that corresponds to the best score, y_best.
        best = np.argwhere(score == np.amax(score))
        y_best=np.median(y0_shifts[best])
        print('y_best ',y_best)
        plt.figure()
        plt.plot(y0_shifts,score,'.');
        plt.plot(np.ones_like(np.arange(1,np.amax(score)+1))*y_best,np.arange(1,np.amax(score)+1),'r')
        plt.title(' score vs y0_shifts(blu), y_best(red)')
        # y0 is the median of these y0 vals                     
        return y_best




   
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