# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 13:06:47 2017

 

@author: ben
"""
import numpy as np
import scipy.sparse as sparse
#import scipy.stats as stats
import scipy.linalg as linalg
from RDE import RDE


class poly_ref_surf:
    def __init__(self, degree_x, degree_y, x0, y0, skip_constant=False, xy_scale=1.0):  
        self.degree_x=degree_x  # degree_x is 'of the regression', ie 0=constant, 1=linear
        self.degree_y=degree_y
        self.x0=x0
        self.y0=y0
        poly_exp_x, poly_exp_y=np.meshgrid(np.arange(0, self.degree_x+1), np.arange(0, self.degree_y+1))
        temp=np.asarray(list(set(zip(poly_exp_x.ravel(), poly_exp_y.ravel()))))  # makes one array out of two
        # sort the coefficients by x degree, then by y degree
        temp=temp[np.sum(temp, axis=1)<=np.maximum(self.degree_x, self.degree_y),:];   # want no non-linear
        # if the skip_constant option is chosen, eliminate the constant term        
        if skip_constant:
            temp=temp[np.all(temp>0, axis=1)]
        # sort the exponents first by x, then by y    
        temp=temp[(temp[:,0]+temp[:,1]/(temp.shape[0]+1.)).argsort()]  # orders the array: 0,0  0,1  1,0
        self.exp_x=temp[:,0]  # [0,0,1]
        self.exp_y=temp[:,1]  # [0,1,0]
        # establish arrays for fitting
        self.poly_vals=np.NaN+np.zeros(self.exp_x.shape)
        self.model_cov_matrix=None
        self.xy_scale=xy_scale
        self.skip_constant=skip_constant
    def fit_matrix(self, x, y):
        G=np.zeros([x.size, self.exp_x.size])  # size is the ravel of shape, here G is len(x) * 3
        for col, ee in enumerate(zip(self.exp_x, self.exp_y)):
            G[:,col]=((x.ravel()-self.x0)/self.xy_scale)**ee[0] * ((y.ravel()-self.y0)/self.xy_scale)**ee[1]
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
        # build a sparse covariance matrix
        if sigma_d is None:
            sigma_d=np.ones_like(xd.ravel())
        chi2r=len(zd)*2
        mask=np.ones_like(zd.ravel(), dtype=bool)
        for k_it in np.arange(max_iterations):
            rows=mask
            sigma_inv=sparse.diags(1/sigma_d.ravel()[rows])
            Gsub=G[rows,:]
            cols=(np.amax(Gsub,axis=0)-np.amin(Gsub,axis=0))>0
            if self.skip_constant is False:
                cols[0]=True
            m=np.zeros([Gsub.shape[1],1])
            # compute the LS coefficients
            msub, rr, rank, sing=linalg.lstsq(sigma_inv.dot(Gsub[:,cols]), sigma_inv.dot(zd.ravel()[rows]))
            msub.shape=(len(msub), 1)
            m[np.where(cols)]=msub  # only takes first three coefficients?
            residual=zd.ravel()-G.dot(m).ravel()
            rs=residual/sigma_d.ravel()
            chi2r_last=chi2r
            chi2r=sum(rs**2)/(len(rows)-len(cols))  
            #print('chi2r is ',chi2r)
            if np.abs(chi2r_last-chi2r)<0.01 or chi2r<1:
                break
            sigma=RDE(rs[rows])
            print('sigma from RDE ',sigma)
            threshold=3.*np.max(sigma, min_sigma)
            mask=np.abs(rs)<threshold
            print "\tsigma=%3.2f, f=%d/%d" % (sigma, np.sum(mask), len(mask))
            # In the future, compute the LS coefficients using PySPQR (get from github.com/yig/PySPQR)
        
        return m, residual, chi2r, rows
