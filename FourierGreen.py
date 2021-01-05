#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 10:15:44 2020

@author: orbitalclover
"""

import numpy as np

def g_hat(Kx, Ky, normKz, normKt, k0, z):
      
    G_hat = np.zeros((*Kx.shape,3,3), dtype = complex);
    
    G_hat[:,:,0,2] = 1j / (2*normKz) * (-Kx*normKz) / k0**2 * np.sign(z) * (np.sqrt(Kx**2+Ky**2)<k0)
    G_hat[:,:,2,0] = G_hat[:,:,0,2]
    
    G_hat[:,:,1,2] = 1j / (2*normKz) * (-Ky*normKz) / k0**2 * np.sign(z) * (np.sqrt(Kx**2+Ky**2)<k0)
    G_hat[:,:,2,1] = G_hat[:,:,1,2]
        
    G_hat[:,:,2,2] = 1j / (2*normKz) * (normKt**2/k0**2) *  (np.sqrt(Kx**2+Ky**2)<k0)
    
    G_hat[:,:,0,1] = 1j / (2*normKz) * (-Ky*Kx) / k0**2 * (np.sqrt(Kx**2+Ky**2)<k0)
    G_hat[:,:,1,0] = G_hat[:,:,0,1]
    
    G_hat[:,:,1,1]  = 1j / (2*normKz) * (1-Ky**2/k0**2) * (np.sqrt(Kx**2+Ky**2)<k0)
    
    G_hat[:,:,0,0]  = 1j / (2*normKz) * (1-Kx**2/k0**2) * (np.sqrt(Kx**2+Ky**2)<k0)
    
    G_hat = G_hat * np.exp(-1j*normKz.reshape((*Kx.shape,1,1))*np.abs(z))
    
    G_hat[np.isnan(G_hat)] = 0
    
    return G_hat

    
def rotg_hat(Kx, Ky, normKz, normKt, k0, z):

    rotG_hat = np.zeros((*Kx.shape,3,3), dtype = complex)

    rotG_hat[:,:,0,2]= - 1/(2*normKz) * Ky * (np.sqrt(Kx**2+Ky**2)<k0)
    rotG_hat[:,:,2,0] = - rotG_hat[:,:,0,2]
    
    rotG_hat[:,:,0,1]  =  1/(2*normKz) * normKz * (np.sqrt(Kx**2+Ky**2)<k0) * np.sign(z)
    rotG_hat[:,:,1,0] = - rotG_hat[:,:,0,1]
    
    rotG_hat[:,:,1,2] = 1/(2*normKz) * Kx * (np.sqrt(Kx**2+Ky**2)<k0)
    rotG_hat[:,:,2,1] = - rotG_hat[:,:,1,2]
    
    rotG_hat = rotG_hat * np.exp(-1j*normKz.reshape((*Kx.shape,1,1))*np.abs(z))
    
    rotG_hat[np.isnan(rotG_hat)] = 0
    
    return rotG_hat


def g_n_hat(G_hat,n):

    G_n_hat = np.zeros(G_hat.shape, dtype = complex)
    
    G_n_hat[:,:,:,0] =  - np.sign(n) * G_hat[:,:,:,1]
    G_n_hat[:,:,:,1] =  np.sign(n) * G_hat[:,:,:,0]
    
    G_n_hat[np.isnan(G_n_hat)] = 0
    
    return G_n_hat 

    
def rotg_n_hat(rotG_hat,n):

    rotG_n_hat = np.zeros(rotG_hat.shape, dtype = complex)
    
    rotG_n_hat[:,:,:,0] =  - np.sign(n) * rotG_hat[:,:,:,1]
    rotG_n_hat[:,:,:,1] =  np.sign(n) * rotG_hat[:,:,:,0]
    
    rotG_n_hat[np.isnan(rotG_n_hat)] = 0
    
    return rotG_n_hat


def propagation(E_hat, H_hat, G_hat, rotG_hat, omega, mu0, eps0):
    
    E2_hat = np.zeros(E_hat.shape, dtype = E_hat.dtype)
    H2_hat = np.zeros(H_hat.shape, dtype = H_hat.dtype)
    
    for ii in range(3):
        for jj in range(3):
            E2_hat[:,:,ii] = E2_hat[:,:,ii] - rotG_hat[:,:,ii,jj] * E_hat[:,:,jj] \
                - 1j *omega * mu0 * G_hat[:,:,ii,jj] * H_hat[:,:,jj]

            H2_hat[:,:,ii] = H2_hat[:,:,ii] - rotG_hat[:,:,ii,jj] * H_hat[:,:,jj] \
                + 1j *omega * eps0 * G_hat[:,:,ii,jj] * E_hat[:,:,jj]

    E2_hat[np.isnan(E2_hat)] = 0
    H2_hat[np.isnan(H2_hat)] = 0
    
    return E2_hat, H2_hat