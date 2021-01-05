#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 17:06:15 2020

@author: orbitalclover
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import FourierGreen as FG
import Aperture as A
import Cavity as C

 
#### Physical parameters
    
eps0 = 8.85418782e-12
mu0 = 1.25663706e-6
c = 1 / np.sqrt(eps0 * mu0)

L = 10e-2

Npix = 20  
Nppix = 4
Npixy = Npix
Npixx = Npix

Lx = L
Ly = L

dx = Lx / (Npixx * Nppix)
dy = Ly / (Npixy * Nppix)

Nx = np.int(2 * Npixx * Nppix)
Ny = np.int(2 * Npixy * Nppix)

x = np.linspace(-(Nx-1)/2,(Nx-1)/2,Nx)*dx
y = np.linspace(-(Ny-1)/2,(Ny-1)/2,Ny)*dy

##### Fourier transform definition

Lambda = 1e-2
k0 = 2 * np.pi / Lambda
omega = c * k0

dkx = 2 * np.pi / (2 * Lx)
dky = 2 * np.pi / (2 * Ly)

Nkx = np.int(np.ceil(k0 / dkx))
Nky = np.int(np.ceil(k0 / dky))

kx = np.linspace(-Nkx, Nkx, 2*Nkx+1) * dkx
ky = np.linspace(-Nky, Nky, 2*Nky+1) * dky

gkyy = np.exp(1j*np.outer(ky[::-1],y[::-1]))
gxkx = np.exp(1j*np.outer(x,kx))
gyky = 1/Ny*np.exp(-1j*np.outer(y[::-1],ky[::-1]))
gkxx = 1/Nx*np.exp(-1j*np.outer(kx,x))

Kx = np.outer(np.ones(2*Nky+1),kx)
Ky = np.outer(ky[-1::-1],np.ones(2*Nkx++1))
normKt = np.sqrt(Kx**2+Ky**2)*(np.sqrt(Kx**2+Ky**2)<k0)
normKz = np.sqrt(k0**2-normKt**2)

H = 1.25e-2
h = 0

Gst_hat = FG.g_hat(Kx, Ky, normKz, normKt, k0, H-h)
rotGst_hat = FG.rotg_hat(Kx, Ky, normKz, normKt, k0, H-h)

Gsb_hat = FG.g_hat(Kx, Ky, normKz, normKt, k0, -H-h)                       
rotGsb_hat = FG.rotg_hat(Kx, Ky, normKz, normKt, k0, -H-h)

Gbt_hat = FG.g_hat(Kx, Ky, normKz, normKt, k0, 2*H)                    
Gbt_hat = FG.g_n_hat(Gbt_hat,1)
rotGbt_hat = FG.rotg_hat(Kx, Ky, normKz, normKt, k0, 2*H)
rotGbt_hat = FG.rotg_n_hat(rotGbt_hat,1)      

Gtb_hat = FG.g_hat(Kx, Ky, normKz, normKt, k0, -2*H)
Gtb_hat = FG.g_n_hat(Gtb_hat,-1)
rotGtb_hat = FG.rotg_hat(Kx, Ky, normKz, normKt, k0, -2*H)
rotGtb_hat = FG.rotg_n_hat(rotGtb_hat,-1)

##### Aperture Matrix

d = 1e-3
eps = 4.3                                                             
Z = - 1j*30/(120*np.pi)                                            
thresh = 1e-12
sgn = 1

[Tmat,Rmat] = A.DielectricLayer(eps,d,Z,kx,ky,k0,sgn,thresh)   

RR = 15

###### Source

Source_field0 = np.zeros((Ny//2,Nx//2,3), dtype = complex) 
Source_field0[Ny//4-1:Ny//4+1,Nx//4-1:Nx//4+1,2] = 1
Source_field = C.symmetrizationJe(Source_field0)

Source_hat = np.zeros((2*Nky+1,2*Nkx+1,3), dtype = complex)
for ii in range(3):
    Source_hat[:,:,ii] = gkyy.dot(Source_field[:,:,ii]).dot(gxkx)
    

##### Initialization / Incident Fields
    
E_top0_hat = np.zeros((2*Nky+1,2*Nkx+1,3), dtype = complex)
H_top0_hat = np.zeros((2*Nky+1,2*Nkx+1,3), dtype = complex)
for ii in range(3):
    for jj in range(3):
        E_top0_hat[:,:,ii] = E_top0_hat[:,:,ii] +  1j * omega * mu0 * Source_hat[:,:,jj] * Gst_hat[:,:,ii,jj]
        H_top0_hat[:,:,ii] = H_top0_hat[:,:,ii]  + Source_hat[:,:,jj] * rotGst_hat[:,:,ii,jj]

        
E_bot0_hat = np.zeros((2*Nky+1,2*Nkx+1,3), dtype = complex)
H_bot0_hat = np.zeros((2*Nky+1,2*Nkx+1,3), dtype = complex)
for ii in range(3):
    for jj in range(3):
        E_bot0_hat[:,:,ii] = E_bot0_hat[:,:,ii] +  1j * omega * mu0 * Source_hat[:,:,jj] * Gsb_hat[:,:,ii,jj]
        H_bot0_hat[:,:,ii] = H_bot0_hat[:,:,ii]  + Source_hat[:,:,jj] * rotGsb_hat[:,:,ii,jj]     
        

##### Configuration and Surface tensors
        
Configuration0 = 2*np.floor(np.random.rand(Npix,Npix)+1/2)-1
Configuration = C.symmetrizationConfig(Configuration0,Npix,Nppix)

SurfaceE = - np.ones((Ny,Nx,3))
SurfaceE[:,:,2] = - SurfaceE[:,:,2]
SurfaceE = - Configuration[:,:,np.newaxis] * SurfaceE

SurfaceH = - SurfaceE

##### 1st reflection on metasurface

E_bot = np.zeros((Ny,Nx,3), dtype = complex)
H_bot = np.zeros((Ny,Nx,3), dtype = complex)
for ii in range(3):
    E_bot[:,:,ii] = gyky.dot(E_bot0_hat[:,:,ii]).dot(gkxx)
    H_bot[:,:,ii] = gyky.dot(H_bot0_hat[:,:,ii]).dot(gkxx)

E_bot_hat = np.zeros((2*Nky+1,2*Nkx+1,3), dtype = complex)
H_bot_hat = np.zeros((2*Nky+1,2*Nkx+1,3), dtype = complex)
for ii in range(3):
    E_bot_hat[:,:,ii] = gkyy.dot((1+SurfaceE[:,:,ii])*E_bot[:,:,ii]).dot(gxkx)
    H_bot_hat[:,:,ii] = gkyy.dot((1+SurfaceH[:,:,ii])*H_bot[:,:,ii]).dot(gxkx)
    

##### Propagation from metasurface to aperture

[E_top_hat, H_top_hat] = FG.propagation(E_bot_hat, H_bot_hat, Gbt_hat, rotGbt_hat, omega, mu0, eps0)

E_top_hat = E_top_hat + E_top0_hat
H_top_hat = H_top_hat + H_top0_hat

E_tot_hat = E_top_hat
H_tot_hat = H_top_hat

##### Reflection on the aperture

Er_hat = np.zeros((2*Nky+1,2*Nkx+1,3), dtype = complex)
Hr_hat = np.zeros((2*Nky+1,2*Nkx+1,3), dtype = complex)
for ii in range(3):
    for jj in range(3):
        Er_hat[:,:,ii] = Er_hat[:,:,ii] + Rmat[:,:,ii,jj] * E_top_hat[:,:,jj]
        Hr_hat[:,:,ii] = Hr_hat[:,:,ii] + np.sqrt(eps0/mu0) * Rmat[:,:,ii+3,jj] * E_top_hat[:,:,jj]
        
E_top_hat = E_top_hat + Er_hat
H_top_hat = H_top_hat + Hr_hat

##### Recursive reflections

for rr in range(RR):
    
    [E_bot_hat, H_bot_hat] = FG.propagation(E_top_hat, H_top_hat, \
                                            Gtb_hat, rotGtb_hat, omega, mu0, eps0)
        
    E_bot = np.zeros((Ny,Nx,3), dtype = complex)
    H_bot = np.zeros((Ny,Nx,3), dtype = complex)
    for ii in range(3):
        E_bot[:,:,ii] = gyky.dot(E_bot_hat[:,:,ii]).dot(gkxx)
        H_bot[:,:,ii] = gyky.dot(H_bot_hat[:,:,ii]).dot(gkxx)

    
    E_bot_hat = np.zeros((2*Nky+1,2*Nkx+1,3), dtype = complex)
    H_bot_hat = np.zeros((2*Nky+1,2*Nkx+1,3), dtype = complex)
    for ii in range(3):
        E_bot_hat[:,:,ii] = gkyy.dot((1+SurfaceE[:,:,ii])*E_bot[:,:,ii]).dot(gxkx)
        H_bot_hat[:,:,ii] = gkyy.dot((1+SurfaceH[:,:,ii])*H_bot[:,:,ii]).dot(gxkx)
        
    [E_top_hat, H_top_hat] = FG.propagation(E_bot_hat, H_bot_hat, Gbt_hat, rotGbt_hat, omega, mu0, eps0)
    
    E_tot_hat = E_tot_hat + E_top_hat
    H_tot_hat = H_tot_hat + H_top_hat
    
    Er_hat = np.zeros((2*Nky+1,2*Nkx+1,3), dtype = complex)
    Hr_hat = np.zeros((2*Nky+1,2*Nkx+1,3), dtype = complex)
    for ii in range(3):
        for jj in range(3):
            Er_hat[:,:,ii] = Er_hat[:,:,ii] + Rmat[:,:,ii,jj] * E_top_hat[:,:,jj]
            Hr_hat[:,:,ii] = Hr_hat[:,:,ii] + np.sqrt(eps0/mu0) * Rmat[:,:,ii+3,jj] * E_top_hat[:,:,jj]
            
    E_top_hat = E_top_hat + Er_hat
    H_top_hat = H_top_hat + Hr_hat

##### Transmission of the field through the aperture
    
Et_hat = np.zeros((2*Nky+1,2*Nkx+1,3), dtype = complex)
Ht_hat = np.zeros((2*Nky+1,2*Nkx+1,3), dtype = complex)
for ii in range(3):
    for jj in range(3):
        Et_hat[:,:,ii] = Et_hat[:,:,ii] + Tmat[:,:,ii,jj] * E_tot_hat[:,:,jj]
        Ht_hat[:,:,ii] = Ht_hat[:,:,ii] + np.sqrt(eps0/mu0) * Tmat[:,:,ii+3,jj] * E_tot_hat[:,:,jj]

Et = np.zeros((Ny,Nx,3), dtype = complex)
Ht = np.zeros((Ny,Nx,3), dtype = complex)

for ii in range(3):
    Et[:,:,ii] = gyky.dot(Et_hat[:,:,ii]).dot(gkxx)
    Ht[:,:,ii] = gyky.dot(Ht_hat[:,:,ii]).dot(gkxx)


for ii in range(3):    
    plt.subplot(2,3,ii+1).imshow(np.abs(Et[:,:,ii]))
    plt.subplot(2,3,ii+4).imshow(np.abs(Ht[:,:,ii]))
plt.show()

