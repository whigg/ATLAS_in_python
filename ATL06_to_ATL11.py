import numpy as np

import scipy.linag as linalg
import scipy.sparse as sparse
import h5py

from ATLAS_land_ice import *


def fit_ATL11(ATL06_files, seg_centers=None):
    params_11=ATL11_params()
    # read in the ATL06 data
    D6=list()  # list of ATL06 files
    for ATL06_file in ATL06_files:
        D6.append(ATL06_data.read_from_file(ATL06_file))
    D6=flatten_struct(D6)
    if seg_centers is None:
        # NO: select every nth center
        seg_centers=np.arange(min(np.c_(D6.x_RGT)), np.max(np.c_[D6.x_RGT]), params_11.seg_RGT_spacing)
    for seg_center in seg_centers:
        D6_sub=D6.index(np.abs(D6.x_RGT-seg_center) < params_11.L_search_AT/2)
        # select by data quality
        selected_pairs, pair_vaild_xslope, pair_valid_yslope, pair_valid_data=select_ATL06_pairs(D6_sub, x_ATC_center, status, ATL11_params)
        # run fit
        
def select_ATL06_pairs(D6, x_ATC_center, status, ATL11_params):
    # this is section 5.1.2: select pairs for reference-surface calculation
    status=dict()    
    seg_valid_data=np.zeros_like(D6_data.time, dtype=np.bool)
    seg_valid_xslope=np.zeros_like(D6.x_ATC, dtype=np.bool)  
    pair_valid_yslope=np.zeros_like(D6_data.time[:,0], dtype=np.bool)
    pair_valid_data=np.zeros_like(D6_data.time[:,0], dtype=np.bool)
    pair_valid_ysearch=np.zeros_like(D6_data.time[:,0], dtype=np.bool)
    
    # step 1a:
    seg_valid_data(np.where(D6.ATL06_quality_summary==0))=True
    # step 1b; the backup step here is UNDOCUMENTED AND UNTESTED
    if not np.any(seg_valid_data):
        status['ATL06_quality_summary_all_nonzero']=1.0
        seg_valid_data(np.where(np.logical_or(ATL06.snr_significance<0.02, ATL06.signal_selection_source <=2)))=True
        if not np.any(seg_valid_data):
            status['ATL06_quality_all_bad']=1
            return seg_valid_data, pair_valid_data, status
            
    seg_sigma_threshold=np.maximum(0.05, 3*np.median(D6.sigma_h_li(np.where(seg_valid_data))))
    status('N_above_data_quality_threshold')=np.sum(D6.sigma_h_li<seg_sigma_threshold)
    seg_valid_data=np.logical_and(seg_valid_data,D6.sigma_h_li<seg_sigma_threshold)
    seg_valid_data=np.logical_and(seg_valid_data, np.all(np.isfinite(D6.sigma_h_li), axis=1))    
    
    # step 1c map seg_valid_data to pair_valid_data
    # Let's assume that ATL06 data fields are Nx2
    pair_valid_data=np.logical_and(seg_valid_data[:,0], seg_valid_data[:,1])
    if not np.any(pair_valid_data):
        status['no_valid_pairs']=1
        return seg_valid_data, pair_valid_data, status
    
    # 2a. define representative x and y values for the valid pairs
    valid_pairs=np.where(pair_valid_data)
    x_pair=np.zeros_like(pair_valid_data)
    y_pair=np.zeros_like(pair_valid_data)
    for pair in valid_pairs:
        x_pair[pair]=np.mean(D6.x_atc[pair,:])
        y_pair[pair]=np.mean(D6.y_atc[pair,:])
    
    # 2b. Calculate the y center of the slope regression
    y_seg_ctr=np.median(y_pair(valid_pairs))
    
    # 2c. identify segments close enough to the y center
    pair_valid_ysearch=np.zeros_like(pair_valid_data, dtype=np.bool)
    pair_valid_ysearch(valid_pairs(np.abs(y_pair(valid_pairs)-y_seg_ctr)<params.L_search_XT/2))
    
    # 3a: combine data and ysearch
    pair_valid_for_y_fit=np.logical_and(pair_valid_data, pair_valid_ysearch)
    
    # 3b:choose the degree of the regression for across-track slope
    if len(set(x_pair[pair_valid_for_y_fit]))>1:
        yslope_regression_x_degree=1
    else:
        yslope_regression_x_degree=0
    if len(set(y_pair[pair_valid_for_y_fit]))>1:
        yslope_regression_y_degree=1
    else:
        yslope_regression_y_degree=0

    #3c: Calculate yslope_regression_tol
    sigma_y_slope=np.sqrt(np.sum(D6.h_li_sigma[pair_valid_for_y_fit,:]**2,axis=1))/np.diff(D6.y_atc[pair_valid_for_y_fit,:],axis=1)
    yslope_regression_tol=np.maximum(0.01, 3*np.median(sigma_y_slope))

    # 3d: regression of yslope against x_pair and y_pair
    slope_regression_y=poly_ref_surf(yslope_regression_x_degree, yslope_regression_y_degree, x_ATC_ctr, y_seg_ctr)
    slope_model_y, slope_y_r,  slope_x2r_y, slope_valid_y_flag=slope_regression_y.fit(pair_x[pair_valid_for_y_fit], pair_y[pair_valid_for_y_fit], D6.dh_fit_dy[pair_valid_for_y_fit,0], max_iterations=2, min_sigma=slope_regression_tol)
    pair_valid_yslope(np.where(pair_valid_for_y_fit))=xtrack_slope_valid_y_flag
    
    #4a. define segs_valid_for_x_fit
    pair_valid_for_x_fit= np.logical_and(pair_valid_data, pair_valid_ysearch)
    
    # 4b:choose the degree of the regression for across-track slope
    if len(set(D6.x_atc[pair_valid_for_x_fit]))>1:
        xslope_regression_x_degree=1
    else:
        xslope_regression_x_degree=0
    if len(set(D6.y_atc[pair_valid_for_y_fit,:].ravel()))>1:
        xslope_regression_y_degree=1
    else:
        xslope_regression_y_degree=0

    #4c: Calculate yslope_regression_tol
    xslope_regression_tol=np.maximum(0.01, 3*np.median(D6.dh_fit_dx_sigma[pair_valid_for_x_fit,:].flatten())) 

    # 4d-4g: regression of xslope against x_pair and y_pair
    slope_regression_x=poly_ref_surf(xslope_regression_x_degree, xslope_regression_y_degree, x_ATC_ctr, y_seg_ctr) 
    slope_model_x, slope_x_r,  slope_x2r_x, slope_valid_x_flag=slope_regression_y.fit(D6.x_ATC[pair_valid_for_x_fit,:].ravel(), D6.y_ATC[pair_valid_for_y_fit,:].ravel(), D6.dh_fit_dx[seg_valid_for_x_fit,:].ravel(), max_iterations=2, min_sigma=xslope_regression_tol)
     
    slope_valid_x_flag.shape=[np.sum(pair_valid_for_x_fit),2]
    seg_valid_xslope[np.where(pair_valid_for_x_fit),:]=slope_valid_x_flag
    pair_valid_xslope=np.all(seg_valid_xslope, axis=1) 
    
    # 5: define selected pairs
    selected_pairs=np.logical_and(pair_valid_data, np.logical_and(pair_valid_xslope, pair_valid_yslope))
    return selected_pairs, pair_vaild_xslope, pair_valid_yslope, pair_valid_data
    

def select_y_center(selected_pairs, unselected_segs, params):
    yvals_for_repeats=dict()
    for pair in selected_pairs:
        yvals_for_repeats[pair.rep_num].append(mean(pair.y_ATC))
    y_rep=np.zeros((len(keys(selected_pairs)),1))
    reps=np.array(keys(selected_pairs))
    for count, rep in enumerate reps:
        y_rep[count]=np.median(np.array(yvals_for_repeats(rep)))
    y0=np.median(y_rep)
    y0_ctrs=np.round(y0)+np.arange(-100,100, 10)
    for y0_ctr, count in enumerate y0_ctrs:
        score[count]=np.sum(np.abs(y_rep-y0_ctr)<params.y_search-params.beam_spacing/2)+len(set(unselected_segs[np.abs(unselected_segs.y_ATC-y0_ctr)<params.y_search].rep))/100
    
    
    

