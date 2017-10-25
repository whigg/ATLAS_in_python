# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 08:50:06 2017

@author: ben
"""
import numpy as np
import scipy.linag as linalg
import scipy.sparse as sparse
import h5py
  


class ATL11_record:
    def __init__(self, N_ref_pts, N_reps, N_epochs):
        self.Data=[]
        # define empty records here based on ATL11 ATBD
        
        
class ATL11_defaults:
    def __init__(self):
        # provide option to read keyword=val pairs from the input file
        self.L_search_AT=120
        self.L_search_XT=120
        self.min_slope_tol=0.02
        self.min_h_tol=0.1


class poly_ref_surf:
    def __init__(self, degree_x, degree_y, x0, y0, skip_constant=False, xy_scale=1.0):
        self.degree_x=degree_x
        self.degree_y=degree_y
        self.x0=x0
        self.y0=y0
        poly_exp_x, poly_exp_y=np.meshgrid(np.arange(0, self.degree_x+1), np.arange(0, self.degree_y+1))
        temp=np.array(list(set(zip(poly_exp_x.ravel(), poly_exp_y.ravel()))))
        # sort the coefficients by x degree, then by y degree
        temp=temp[np.where(np.sum(temp, axis=1)<=np.maximum(self.degree_x, self.degree_y)),:];  
        # if the skip_constant option is chosen, eliminate the constant term        
        if skip_constant:
            temp=temp(np.all(temp>0, axis=1))
        # sort the exponents first by x, then by y    
        temp=temp[(temp[:,0]+temp[:,1]/(temp.shape[0]+1)).argsort()]
        self.exp_x=temp[:,0]
        self.exp_y=temp[:,1]
        self.poly_vals=np.NaN(self.exp_x.shape)
        self.model_cov_matrix=None
        self.xy_scale=xy_scale
    def fit_matrix(self, x, y):
        G=np.zeros([x.size, self.exp_x.size])
        for col, ee in enumerate(zip(self.exp_x, self.exp_y)):
            G[:,col]=(x.ravel()/self.xy_scale)**ee[0] * (y.ravel()/self.xy_scale)**ee[1]
        return G
    def z(self, x0, y0):
        # evaluate the poltnomial at [x0, y0]
        G=self.fit_matrix(x0, y0)
        z=np.dot(G, self.poly_vals)
        z.shape=x0.shape
        return z
    def fit(self, xd, yd, zd, sigma_d=None, max_iterations=1, min_sigma=0):
        # asign poly_vals and cov_matrix with a linear fit to zd at points xd, yd
        # build the design matrix:      
        G=self.fit_matrix(xd, yd)
        N_data=xd.ravel().shape[0]
        # build a sparse covariance matrix
        if sigma_d is None:
            sigma_d=np.ones_like(xd.ravel())
        X2r=len(zd)*2
        mask=np.ones_like(zd.ravel(), dtype=bool)
        for k_it in np.arange(max_iterations):
            rows=np.where(mask)
            sigma_inv=sparse.diags(1/sigma_d[rows])
            Gsub=G[rows,:]
            cols=np.c_[0,np.where([np.max(Gsub,0)-np.min(Gsub,0)]>0)]
            m=np.zeros(1, Gsub.shape[1])
            # compute the LS coefficients
            msub, rr, rank, sing=linalg.lstsq(sigma_inv.dot(Gsub[:,cols]), sigma_inv.dot(zd[rows]))
            m[cols]=msub
            r=zd.ravel()-m.dot(G)
            rs=sigma_inv.dot(r[rows])
            X2r_last=X2r
            X2r=sum(rs**2)/(len(rows)-len(cols))       
            if np.abs(X2r_last-X2r)<0.01 or X2r<1:
                break
            sigma=RDE(rs)
            threshold=3.*np.max(sigma, min_sigma)
            mask=np.abs(r)<threshold
            # In the future, compute the LS coefficients using PySPQR (get from github.com/yig/PySPQR)
        
        return m, r, X2r


def RDE(x):
    xs=x.copy()
    xs=xs(np.isfinite(xs))
    if len(xs)<2 :
        return np.nan
    ind=np.arange(0.5, len(xs))
    LH=np.interp(np.array([0.16, 0.84])*len(xs), ind, xs.sorted())
    return (LH[1]-LH[0])/2.

   
            

def flatten_struct(a):
    b=a[0].__init__
    if len(a)==1:
        b=a[0].copy()
        return b
    fields=a.dict().keys()
    for field in fields:
        b.setattr(field, np.concatenate([x.getattr(field) for x in a] ))
    return b
