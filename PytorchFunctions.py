# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 12:18:25 2020

@author: Greenerwave
"""

import torch
import numpy as np

def numpytotorch(A, device):
    
    A_tensor  = dict()
    A_tensor['r'] = torch.from_numpy(np.real(A)).type(torch.FloatTensor).to(device)
    A_tensor['i'] = torch.from_numpy(np.imag(A)).type(torch.FloatTensor).to(device)
    
    return A_tensor

def sumcomplex(A,B):
    
    C  = dict()
    C['r'] = A['r'] + B['r']
    C['i'] = A['i'] + B['i']
    
    return C

def matmulcomplex(A,B):
    
    C  = dict()
    C['r'] = torch.matmul(A['r'],B['r']) - torch.matmul(A['i'],B['i'])  
    C['i'] = torch.matmul(A['r'],B['i']) + torch.matmul(A['i'],B['r'])  
    
    return C


def torchFourier(A,g1,g2,bs = 1):
    
    if bs == 1:
        
        C = dict()
        C['r'] = torch.zeros((g1['r'].shape[0],g2['r'].shape[1],3), device = g1['r'].device)
        C['i'] = torch.zeros((g1['r'].shape[0],g2['r'].shape[1],3), device = g1['r'].device)
        
        for ii in range(3):
            
            C['r'][:,:,ii] = torch.matmul(g1['r'], torch.matmul(A['r'][:,:,ii],g2['r'])) - \
                            torch.matmul(g1['r'], torch.matmul(A['i'][:,:,ii],g2['i'])) - \
                            torch.matmul(g1['i'], torch.matmul(A['r'][:,:,ii],g2['i'])) - \
                            torch.matmul(g1['i'], torch.matmul(A['i'][:,:,ii],g2['r'])) 
                            
            C['i'][:,:,ii] = torch.matmul(g1['r'], torch.matmul(A['r'][:,:,ii],g2['i'])) + \
                            torch.matmul(g1['r'], torch.matmul(A['i'][:,:,ii],g2['r'])) + \
                            torch.matmul(g1['i'], torch.matmul(A['r'][:,:,ii],g2['r'])) - \
                            torch.matmul(g1['i'], torch.matmul(A['i'][:,:,ii],g2['i'])) 
        
    else:
        
        C = dict()
        C['r'] = torch.zeros((bs,g1['r'].shape[0],g2['r'].shape[1],3), device = g1['r'].device)
        C['i'] = torch.zeros((bs,g1['r'].shape[0],g2['r'].shape[1],3), device = g1['r'].device)
            
        
        if A['r'].shape[0] == bs:
               
            for ii in range(3):
                
                C['r'][:,:,:,ii] = torch.matmul(g1['r'], torch.matmul(A['r'][:,:,:,ii],g2['r'])) - \
                                torch.matmul(g1['r'], torch.matmul(A['i'][:,:,:,ii],g2['i'])) - \
                                torch.matmul(g1['i'], torch.matmul(A['r'][:,:,:,ii],g2['i'])) - \
                                torch.matmul(g1['i'], torch.matmul(A['i'][:,:,:,ii],g2['r'])) 
                                
                C['i'][:,:,:,ii] = torch.matmul(g1['r'], torch.matmul(A['r'][:,:,:,ii],g2['i'])) + \
                                torch.matmul(g1['r'], torch.matmul(A['i'][:,:,:,ii],g2['r'])) + \
                                torch.matmul(g1['i'], torch.matmul(A['r'][:,:,:,ii],g2['r'])) - \
                                torch.matmul(g1['i'], torch.matmul(A['i'][:,:,:,ii],g2['i'])) 
                                
        else:
            
            for ii in range(3):
                
                C['r'][:,:,:,ii] = torch.matmul(g1['r'], torch.matmul(A['r'][:,:,ii],g2['r'])) - \
                                torch.matmul(g1['r'], torch.matmul(A['i'][:,:,ii],g2['i'])) - \
                                torch.matmul(g1['i'], torch.matmul(A['r'][:,:,ii],g2['i'])) - \
                                torch.matmul(g1['i'], torch.matmul(A['i'][:,:,ii],g2['r'])) 
                                
                C['i'][:,:,:,ii] = torch.matmul(g1['r'], torch.matmul(A['r'][:,:,ii],g2['i'])) + \
                                torch.matmul(g1['r'], torch.matmul(A['i'][:,:,ii],g2['r'])) + \
                                torch.matmul(g1['i'], torch.matmul(A['r'][:,:,ii],g2['r'])) - \
                                torch.matmul(g1['i'], torch.matmul(A['i'][:,:,ii],g2['i'])) 
                                                    
    return C


def IncidentField(Source,G,rotG,eps0,mu0,omega):
    
    E = dict()
    E['r'] = torch.zeros((Source['r'].shape[0],Source['r'].shape[1],3), device = Source['r'].device)
    E['i'] = torch.zeros((Source['r'].shape[0],Source['r'].shape[1],3), device = Source['r'].device)
    H = dict()
    H['r'] = torch.zeros((Source['r'].shape[0],Source['r'].shape[1],3), device = Source['r'].device)
    H['i'] = torch.zeros((Source['r'].shape[0],Source['r'].shape[1],3), device = Source['r'].device)

    for ii in range(3):
        
        for jj in range(3):
            
            E['r'][:,:,ii] = E['r'][:,:,ii] - omega * mu0 * (Source['r'][:,:,jj] * G['i'][:,:,ii,jj] + \
                                                             Source['i'][:,:,jj] * G['r'][:,:,ii,jj])
                
            E['i'][:,:,ii] = E['i'][:,:,ii] + omega * mu0 * (Source['r'][:,:,jj] * G['r'][:,:,ii,jj] - \
                                                             Source['i'][:,:,jj] * G['i'][:,:,ii,jj])
                
            H['r'][:,:,ii] = H['r'][:,:,ii] + (Source['r'][:,:,jj] * rotG['r'][:,:,ii,jj] - \
                                                             Source['i'][:,:,jj] * rotG['i'][:,:,ii,jj])
                
            H['i'][:,:,ii] = H['i'][:,:,ii] + (Source['r'][:,:,jj] * rotG['i'][:,:,ii,jj] + \
                                                             Source['i'][:,:,jj] * rotG['r'][:,:,ii,jj])
            
    return E, H


def symmetrization(Configuration0,Npix,Nppix,bs):
    
    Ny = Nppix*Npix
    Nx = Nppix*Npix

    Configuration = torch.zeros((bs,2*Ny,2*Nx), dtype = Configuration0.dtype, device = Configuration0.device)
    
    for bb in range(bs):
        for ii in range(Npix):
            for jj in range(Npix):
                Configuration[bb,ii*Nppix:(ii+1)*Nppix,jj*Nppix:(jj+1)*Nppix] = Configuration0[bb,ii,jj]
    
    Configuration[:,Ny:,:Nx] = torch.flip(Configuration[:,:Ny,:Nx],[1])
    Configuration[:,:Ny,Nx:] = torch.flip(Configuration[:,:Ny,:Nx],[2])
    Configuration[:,Ny:,Nx:] = torch.flip(Configuration[:,:Ny,:Nx],[1,2])
    
    return Configuration


def reflectionSurface(Field, Surface):
    
    Field2 = dict()
    Field2['r'] = Field['r'] * Surface
    Field2['i'] = Field['i'] * Surface
    
    return Field2


def propagation(E, H, G, rotG, omega, mu0, eps0):
    
    E2 = dict()
    E2['r'] = torch.zeros(E['r'].shape, device = E['r'].device)
    E2['i'] = torch.zeros(E['r'].shape, device = E['r'].device)
    H2 = dict()
    H2['r'] = torch.zeros(H['r'].shape, device = H['r'].device)
    H2['i'] = torch.zeros(H['r'].shape, device = H['r'].device)
    
    for ii in range(3):
        
        for jj in range(3):
            
            E2['r'][:,:,:,ii] = E2['r'][:,:,:,ii] - (rotG['r'][:,:,ii,jj] * E['r'][:,:,:,jj] - \
                                                 rotG['i'][:,:,ii,jj] * E['i'][:,:,:,jj]) \
                                                + omega * mu0 * (G['i'][:,:,ii,jj] * H['r'][:,:,:,jj] + \
                                                                 G['r'][:,:,ii,jj] * H['i'][:,:,:,jj])
                                    
            E2['i'][:,:,:,ii] = E2['i'][:,:,:,ii] - (rotG['r'][:,:,ii,jj] * E['i'][:,:,:,jj] + \
                                                 rotG['i'][:,:,ii,jj] * E['r'][:,:,:,jj]) \
                                                - omega * mu0 * (G['r'][:,:,ii,jj] * H['r'][:,:,:,jj] - \
                                                                 G['i'][:,:,ii,jj] * H['i'][:,:,:,jj])
                                    
            H2['r'][:,:,:,ii] = H2['r'][:,:,:,ii] - (rotG['r'][:,:,ii,jj] * H['r'][:,:,:,jj] - \
                                                 rotG['i'][:,:,ii,jj] * H['i'][:,:,:,jj]) \
                                                - omega * eps0 * (G['i'][:,:,ii,jj] * E['r'][:,:,:,jj] + \
                                                                  G['r'][:,:,ii,jj] * E['i'][:,:,:,jj])
                                    
            H2['i'][:,:,:,ii] = H2['i'][:,:,:,ii] - (rotG['r'][:,:,ii,jj] * H['i'][:,:,:,jj] + \
                                                  rotG['i'][:,:,ii,jj] * H['r'][:,:,:,jj]) \
                                                + omega * eps0 * (G['r'][:,:,ii,jj] * E['r'][:,:,:,jj] - \
                                                                  G['i'][:,:,ii,jj] * E['i'][:,:,:,jj])

    
    return E2, H2

def RefleTransAperture(E, RT, mu0, eps0):
    
    E2 = dict()
    E2['r'] = torch.zeros(E['r'].shape, device = E['r'].device)
    E2['i'] = torch.zeros(E['r'].shape, device = E['r'].device)
    H2 = dict()
    H2['r'] = torch.zeros(E['r'].shape, device = E['r'].device)
    H2['i'] = torch.zeros(E['r'].shape, device = E['r'].device)
    
    for ii in range(3):
        
        for jj in range(3):
            
            E2['r'][:,:,:,ii] = E2['r'][:,:,:,ii] + (RT['r'][:,:,ii,jj] * E['r'][:,:,:,jj] - \
                                                 RT['i'][:,:,ii,jj] * E['i'][:,:,:,jj])
                                    
            E2['i'][:,:,:,ii] = E2['i'][:,:,:,ii] + (RT['r'][:,:,ii,jj] * E['i'][:,:,:,jj] + \
                                                 RT['i'][:,:,ii,jj] * E['r'][:,:,:,jj])
                                    
            H2['r'][:,:,:,ii] = H2['r'][:,:,:,ii] + np.sqrt(eps0/mu0) * (RT['r'][:,:,ii+3,jj] * E['r'][:,:,:,jj] - \
                                                 RT['i'][:,:,ii+3,jj] * E['i'][:,:,:,jj])
                                    
            H2['i'][:,:,:,ii] = H2['i'][:,:,:,ii] + np.sqrt(eps0/mu0) * (RT['r'][:,:,ii+3,jj] * E['i'][:,:,:,jj] + \
                                                 RT['i'][:,:,ii+3,jj] * E['r'][:,:,:,jj])
    
    return E2, H2
    