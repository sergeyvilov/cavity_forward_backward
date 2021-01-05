#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 17:06:15 2020

@author: orbitalclover
"""

import time
import torch
import matplotlib.pyplot as plt
import numpy as np
import FourierGreen as FG
import Aperture as A
import Cavity as C
import PytorchFunctions as PF

#### CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

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

gkyy = PF.numpytotorch(np.exp(1j*np.outer(ky[::-1],y[::-1])), device)
gxkx = PF.numpytotorch(np.exp(1j*np.outer(x,kx)), device)
gyky = PF.numpytotorch(1/Ny*np.exp(-1j*np.outer(y[::-1],ky[::-1])), device)
gkxx = PF.numpytotorch(1/Nx*np.exp(-1j*np.outer(kx,x)), device)

Kx = np.outer(np.ones(2*Nky+1),kx)
Ky = np.outer(ky[-1::-1],np.ones(2*Nkx++1))
normKt = np.sqrt(Kx**2+Ky**2)*(np.sqrt(Kx**2+Ky**2)<k0)
normKz = np.sqrt(k0**2-normKt**2)

H = 1.25e-2
h = 0

Gst_hat = PF.numpytotorch(FG.g_hat(Kx, Ky, normKz, normKt, k0, H-h), device)
rotGst_hat = PF.numpytotorch(FG.rotg_hat(Kx, Ky, normKz, normKt, k0, H-h), device)

Gsb_hat = PF.numpytotorch(FG.g_hat(Kx, Ky, normKz, normKt, k0, -H-h), device)                 
rotGsb_hat = PF.numpytotorch(FG.rotg_hat(Kx, Ky, normKz, normKt, k0, -H-h), device)

Gbt_hat = FG.g_hat(Kx, Ky, normKz, normKt, k0, 2*H)                    
Gbt_hat = PF.numpytotorch(FG.g_n_hat(Gbt_hat,1), device)
rotGbt_hat = FG.rotg_hat(Kx, Ky, normKz, normKt, k0, 2*H)
rotGbt_hat = PF.numpytotorch(FG.rotg_n_hat(rotGbt_hat,1), device)     

Gtb_hat = FG.g_hat(Kx, Ky, normKz, normKt, k0, -2*H)
Gtb_hat = PF.numpytotorch(FG.g_n_hat(Gtb_hat,-1), device)
rotGtb_hat = FG.rotg_hat(Kx, Ky, normKz, normKt, k0, -2*H)
rotGtb_hat = PF.numpytotorch(FG.rotg_n_hat(rotGtb_hat,-1), device)

##### Aperture Matrix

d = 1e-3
eps = 4.3                                                             
Z = - 1j*30/(120*np.pi)                                            
thresh = 1e-12
sgn = 1

[Tmat,Rmat] = A.DielectricLayer(eps,d,Z,kx,ky,k0,sgn,thresh)
Tmat = PF.numpytotorch(Tmat, device)
Rmat = PF.numpytotorch(Rmat, device)

RR = 15

###### Source

Source_field0 = np.zeros((Ny//2,Nx//2,3)) 
Source_field0[Ny//4-1:Ny//4+1,Nx//4-1:Nx//4+1,2] = 1
Source_field = PF.numpytotorch(C.symmetrizationJe(Source_field0), device)

Source_hat = PF.torchFourier(Source_field,gkyy,gxkx)
    
##### Initialization / Incident Fields
    
E_top0_hat, H_top0_hat =  PF.IncidentField(Source_hat,Gst_hat,rotGst_hat,eps0,mu0,omega)
E_bot0_hat, H_bot0_hat =  PF.IncidentField(Source_hat,Gsb_hat,rotGsb_hat,eps0,mu0,omega)
 
##### Configuration and Surface tensors

tic = time.time()

bs = 100

Configuration0 = (2*torch.floor(torch.rand(bs,Npix,Npix)+1/2)-1).to(device)
Configuration = PF.symmetrization(Configuration0, Npix, Nppix, bs)

SurfaceE = - torch.ones((bs,Ny,Nx,3)).to(device)
SurfaceE[:,:,:,2] = - SurfaceE[:,:,:,2]
SurfaceE = - Configuration.unsqueeze(3) * SurfaceE

SurfaceH = - SurfaceE

##### 1st reflection on metasurface

E_bot = PF.torchFourier(E_bot0_hat,gyky,gkxx,bs)
H_bot = PF.torchFourier(H_bot0_hat,gyky,gkxx,bs)

E_bot = PF.reflectionSurface(E_bot,SurfaceE)
H_bot = PF.reflectionSurface(H_bot,SurfaceH)

E_bot_hat = PF.torchFourier(E_bot,gkyy,gxkx,bs)
H_bot_hat = PF.torchFourier(H_bot,gkyy,gxkx,bs)    

##### Propagation from metasurface to aperture

[E_top_hat, H_top_hat] = PF.propagation(E_bot_hat, H_bot_hat, Gbt_hat, rotGbt_hat, omega, mu0, eps0)

E_top_hat = PF.sumcomplex(E_top_hat, E_top0_hat)
H_top_hat = PF.sumcomplex(H_top_hat, H_top0_hat)

E_tot_hat = E_top_hat
H_tot_hat = H_top_hat

##### Reflection on the aperture

[Er_hat, Hr_hat] = PF.RefleTransAperture(E_top_hat, Rmat, mu0, eps0)
 
E_top_hat = PF.sumcomplex(E_top_hat, Er_hat)
H_top_hat = PF.sumcomplex(H_top_hat, Hr_hat)

##### Recursive reflections

for rr in range(RR):
    
    [E_bot_hat, H_bot_hat] = PF.propagation(E_top_hat, H_top_hat, Gtb_hat, rotGtb_hat, omega, mu0, eps0)

    ### reflection on metasurface

    E_bot = PF.torchFourier(E_bot_hat,gyky,gkxx,bs)
    H_bot = PF.torchFourier(H_bot_hat,gyky,gkxx,bs)
    
    E_bot = PF.reflectionSurface(E_bot,SurfaceE)
    H_bot = PF.reflectionSurface(H_bot,SurfaceH)
    
    E_bot_hat = PF.torchFourier(E_bot,gkyy,gxkx,bs)
    H_bot_hat = PF.torchFourier(H_bot,gkyy,gxkx,bs)  

    [E_top_hat, H_top_hat] = PF.propagation(E_bot_hat, H_bot_hat, Gbt_hat, rotGbt_hat, omega, mu0, eps0) 
    
    E_tot_hat = PF.sumcomplex(E_tot_hat, E_top_hat)
    H_tot_hat = PF.sumcomplex(H_tot_hat, H_top_hat)
    
    [Er_hat, Hr_hat] = PF.RefleTransAperture(E_top_hat, Rmat, mu0, eps0)
      
    E_top_hat = PF.sumcomplex(E_top_hat, Er_hat)
    H_top_hat = PF.sumcomplex(H_top_hat, Hr_hat) 
    
#### Transmission of the field through the aperture
    
[Et_hat, Ht_hat] = PF.RefleTransAperture(E_tot_hat, Tmat, mu0, eps0)

Et = PF.torchFourier(Et_hat,gyky,gkxx,bs)
Ht = PF.torchFourier(Ht_hat,gyky,gkxx,bs)

toc = time.time()
    
#Et = Et['r'].cpu().numpy() + 1j * Et['i'].cpu().numpy()
#Ht = Ht['r'].cpu().numpy() + 1j * Ht['i'].cpu().numpy()

#for ii in range(bs):    
#    plt.subplot(2,bs//2,ii+1)
#    plt.imshow(np.abs(Et[ii,:,:,0]))
#plt.show()

# for ii in range(3):    
#     plt.subplot(2,3,ii+1)
#     plt.imshow(np.abs(Et[:,:,ii]))
#     plt.colorbar()
#     plt.subplot(2,3,ii+4)
#     plt.imshow(np.abs(Ht[:,:,ii]))
#     plt.colorbar()
# plt.show()
