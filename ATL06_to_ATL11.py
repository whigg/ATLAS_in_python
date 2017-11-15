import numpy as np


from ATL06_data import ATL06_data

from ATL11 import ATL11_data, ATL11_point, ATL11_defaults
from poly_ref_surf import poly_ref_surf

def fit_ATL11(ATL06_files, pair=1, seg_x_centers=None, output_file=None):
    params_11=ATL11_defaults()
    # read in the ATL06 data
    D6=ATL06_data(filename=ATL06_files, pair=pair)
    if seg_x_centers is None:
        # NO: select every nth center
        seg_x_centers=np.arange(min(np.c_(D6.x_atc)), np.max(np.c_[D6.x_atc]), params_11.seg_atc_spacing)
    for seg_x_center in seg_x_centers:
        D6_sub=D6.subset(np.any(np.abs(D6.x_atc-seg_x_center) < params_11.L_search_AT, axis=1))
        #2a. define representative x and y values for the pairs
        pair_data=D6_sub.get_pairs()
        P11=ATL11_point(N_pairs=len(pair_data.x), x_atc_ctr=seg_x_center, y_atc_ctr=None, track_azimuth=np.nanmedian(D6_sub.seg_azimuth.ravel()) )
        # step 2: select pairs
        P11.select_ATL06_pairs(D6_sub, pair_data, seg_x_center,  params_11)
        #P11.select_y_center(D6_sub)
        #P11.fit_reference_surface(D6_sub, params_11)
        #P11.correct_unpaired_segments(D6_sub)
        # build the ATL11 data structure
        #D11.append_point(P11)
        
        #D6_sub.plot(valid_pairs=P11.valid_pairs.all)
    # D11.write_to_file(output_file)
    return
  
 
    

