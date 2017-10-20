# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:10:32 2017

@author: ben
"""



# poly_ref_surf test
import numpy as np
from poly_ref_surf import poly_ref_surf


degree_x=3
degree_y=3
x0=0
y0=0

xg, yg=np.meshgrid(np.arange(-1, 1, .1), np.arange(-1, 1, .1), dtype='float');

poly_coeffs=[1., 2., 3., 4., 5., 6]
poly_deg_x=[0, 1, 0, 1, 2, 0]
poly_deg_y=[0, 0, 1, 1, 0, 2]

zg=np.zeros_like(xg)
for ii in np.arange(len(poly_coeffs)):
    zg=zg+poly_coeffs[ii]*(xg**poly_deg_x[ii])*(yg**poly_deg_y[ii])
    
P=poly_ref_surf(degree_x, degree_y, x0, y0)

m=P.fit(xg, yg, zg)
for k, ij in enumerate(zip(P.exp_x, P.exp_y)):
    this=np.where(np.logical_and(poly_deg_x==ij[0], poly_deg_y==ij[1]))
    print('%d %d %f %f\n', ij, poly_coeffs[k], m[k] )


