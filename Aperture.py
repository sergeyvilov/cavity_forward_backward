#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 11:06:20 2020

@author: orbitalclover
"""

import numpy as np
from scipy.linalg import expm, block_diag

def DielectricLayer(eps,d,Z,kx,ky,k0,sgn,thresh):
    
    
    RR = np.zeros((len(ky), len(kx), 6, 3), dtype = complex)
    TT = np.zeros((len(ky), len(kx), 6, 3), dtype = complex)
    
    for kkx in zip(*np.where(np.abs(kx)<k0)):
        
        
        for kky in zip(*np.where((ky**2+kx[kkx]**2)<k0**2*(1-thresh))):
                                 
            k_inc = np.array([kx[kkx], -ky[kky], sgn*np.sqrt(k0**2-kx[kkx]**2-ky[kky]**2)])
            
            
            ###internal definitions

            q = np.array([0,0,1])

            qx2 = np.array([[0,-1],[1,0]])
            
            qx = np.array([[0,-1,0],[1,0,0],[0,0,0]])
            
            b = np.array([k_inc[0],k_inc[1],0]) / k0
            
            b2 = np.array([k_inc[0],k_inc[1]]) / k0
            
            a2 = np.array([k_inc[1],-k_inc[0]]) / k0
            
            gamma = np.array([[(k_inc[0]**2+k_inc[2]**2)/k_inc[2], k_inc[0]*k_inc[1]/k_inc[2]], \
                              [k_inc[0]*k_inc[1]/k_inc[2],(k_inc[1]**2+k_inc[2]**2)/k_inc[2]]])/k0
                
                 
            ###The matrix M appearing below describes the slab of the material
            
            v2 = -a2
            
            M = np.zeros((4,4))
            M[0:2,2:] = eps*np.eye(2) + np.outer(b2,qx2.dot(v2))
            M[2:,0:2] = np.eye(2)- np.outer(a2,a2)/eps
           
            ###INVERSE operator of spatial evolution
            ###The first term in P corresponds to the array of patches and the
            ###second one describes wave propagation in the dielectric slab

            ZTE = Z
            ZTM = ZTE
            
            ###
            
            Ps = np.zeros((4,4), dtype = complex)
            Ps[0:2,0:2] = np.eye(2)
            Ps[0:2,2:] = qx2.dot(np.diag([1/ZTE,1/ZTM]).dot(qx2))

            Ps[2:,0:2] = np.zeros((2,2))
            Ps[2:,2:]  = np.eye(2)
            
            P = np.linalg.inv(Ps).dot(expm(1j*sgn*k0*d*M))
                  

            ###calculation of the transmission and reflection operators, initially they
            ### act on the tangential components of the MAGNETIC field H
   
            T = 2 * np.linalg.inv(np.hstack((gamma,np.eye(2))).dot(P).dot(np.vstack((np.eye(2),gamma)))).dot(gamma)
            
            R = -np.eye(2) + np.hstack((np.eye(2),np.zeros((2,2)))).dot(P).dot(np.vstack((np.eye(2),gamma))).dot(T)

            ###operator V performs transformation from tangenetial fields to 3D fields
            
            V = np.vstack((np.hstack((block_diag(np.eye(2),0),np.outer(q,np.cross(b,q)))), \
                           np.hstack((np.outer(q,np.cross(q,b)),block_diag(np.eye(2),0)))))

            ###calculation of the reflected field

            RR[kky,kkx,:,:] = V.dot(np.vstack((qx.dot(block_diag(gamma.dot(R),0)),block_diag(R,0)))) \
                .dot(block_diag(np.linalg.inv(gamma),0)).dot(qx)

            TT[kky,kkx,:,:] = V.dot(np.vstack((-qx.dot(block_diag(gamma.dot(T),0)),block_diag(T,0)))) \
                .dot(block_diag(np.linalg.inv(gamma),0)).dot(qx)

    RR[np.isnan(RR)] = 0
    TT[np.isnan(TT)] = 0
        
    return TT, RR