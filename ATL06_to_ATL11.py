import numpy as np


from ATL06_data import ATL06_data

from ATL11 import ATL11_data, ATL11_point, ATL11_defaults
import matplotlib.pyplot as plt 
import time

#from poly_ref_surf import poly_ref_surf

def fit_ATL11(ATL06_files, beam_pair=1, seg_x_centers=None, output_file=None):
    params_11=ATL11_defaults()
    # read in the ATL06 data
    D6=ATL06_data(filename=ATL06_files, beam_pair=beam_pair)
#    plt.plot(D6.delta_time,D6.h_li,'b.') 
#    filesplt=ATL06_files[0].split('/')[-1]
#    titlestr='Heights of all cycles in files: %s' % (filesplt)
#    plt.title(titlestr)

    if seg_x_centers is None:
        # NO: select every nth center        
        seg_x_centers=np.arange(np.min(np.c_[D6.x_atc]), np.max(np.c_[D6.x_atc]), params_11.seg_atc_spacing)
    for seg_x_center in seg_x_centers:
        D6_sub=D6.subset(np.any(np.abs(D6.x_atc-seg_x_center) < params_11.L_search_AT, axis=1)) 
        
        #2a. define representative x and y values for the pairs
        pair_data=D6_sub.get_pairs()   # this might go, similar to D6_sub
        # pair_data and D6_sub are the same length, but pair_data.y = mean(D6_sub.y_atc) and pair_data.delta_time = mean(D6_sub.delta_time) They are different
        P11=ATL11_point(N_pairs=len(pair_data.x), x_atc_ctr=seg_x_center, y_atc_ctr=None, track_azimuth=np.nanmedian(D6_sub.seg_azimuth.ravel()) )
        # step 2: select pairs, based on reasonable slopes
        P11.select_ATL06_pairs(D6_sub, pair_data, seg_x_center,  params_11)
        P11.y_best=P11.select_y_center(D6_sub, pair_data, params_11)

#        plt.plot(D6_sub.delta_time[P11.valid_pairs.data,:],D6_sub.h_li[P11.valid_pairs.data,:],'ro')
#        plt.plot(D6_sub.delta_time[P11.valid_pairs.x_slope,:],D6_sub.h_li[P11.valid_pairs.x_slope,:],'kx')
#        plt.plot(D6_sub.delta_time[P11.valid_pairs.y_slope.ravel(),:],D6_sub.h_li[P11.valid_pairs.y_slope.ravel(),:],'mD')
#        plt.plot(D6_sub.delta_time[P11.valid_pairs.all,:],D6_sub.h_li[P11.valid_pairs.all,:],'gs')
#        plt.title('Heights subset: init(b),qual(r),dhdx(k),dhdy(m),all(g)')
        
        plt.figure()
        plt.plot(D6_sub.x_atc,'.')
        plt.plot(D6_sub.x_atc[P11.valid_pairs.all,:],'o')
        plt.plot(seg_x_center*np.ones_like(D6_sub.x_atc[:,0]),'r')
        plt.title('D6_sub.x_atc(.), D6_sub.x_atc valid(o), seg_x_center(red)')
        
        plt.figure()
        plt.plot(D6_sub.y_atc,'.')
        plt.plot(D6_sub.y_atc[P11.valid_pairs.all,:],'o')
        plt.plot(P11.y_polyfit_ctr*np.ones_like(D6_sub.y_atc[:,0]),'r--')
        plt.plot(P11.y_best*np.ones_like(D6_sub.y_atc[:,0]),'r')
        plt.title('D6_sub.y_atc(.),D6_sub.y_atc valid(o), y_polyfit_ctr(r--), y_best(red)')
        
        plt.figure()
        plt.plot(D6_sub.cycle,'.')
        plt.plot(D6_sub.cycle[P11.valid_pairs.all,:],'o')
        plt.title('D6_sub.cycle(.) D6_sub.cycle valid(o)')
        #P11.fit_reference_surface(D6_sub, params_11)
        #P11.correct_unpaired_segments(D6_sub)
        # build the ATL11 data structure
        #D11.append_point(P11)
        
        #D6_sub.plot(valid_pairs=P11.valid_pairs.all)
    #plt.plot(P11.)
    # D11.write_to_file(output_file)
    return
  
 
    

