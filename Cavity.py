#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 17:15:43 2020

@author: orbitalclover
"""

import numpy as np

def symmetrizationJe(Field0):
    
    Ny0,Nx0 = Field0.shape[0:2]
    
    Field = np.zeros((2*Ny0,2*Nx0,3),dtype = Field0.dtype)
    
    Field[:Ny0,:Nx0,0] = Field0[:,:,0] 
    Field[Ny0:,:Nx0,0] = - Field0[:,:,0] 
    Field[:Ny0,Nx0:,0] = Field0[:,:,0]
    Field[Ny0:,Nx0:,0] = - Field0[:,:,0]  
    
    Field[:Ny0,:Nx0,1] = Field0[:,:,1] 
    Field[Ny0:,:Nx0,1] = Field0[:,:,1] 
    Field[:Ny0,Nx0:,1] = - Field0[:,:,1]
    Field[Ny0:,Nx0:,1] = - Field0[:,:,1]  
    
    Field[:Ny0,:Nx0,2] = Field0[:,:,2] 
    Field[Ny0:,:Nx0,2] = - Field0[:,:,2] 
    Field[:Ny0,Nx0:,2] = - Field0[:,:,2]
    Field[Ny0:,Nx0:,2] = Field0[:,:,2]  
    
    return Field

def symmetrizationConfig(Configuration0,Npix,Nppix):
    
    Ny = Nppix*Npix
    Nx = Nppix*Npix

    Configuration = np.zeros((2*Ny,2*Nx), dtype = Configuration0.dtype)
    
    for ii in range(Npix):
        for jj in range(Npix):
            Configuration[ii*Nppix:(ii+1)*Nppix,jj*Nppix:(jj+1)*Nppix] = Configuration0[ii,jj]
    
    Configuration[Ny:,:Nx] = np.flipud(Configuration[:Ny,:Nx])
    Configuration[:Ny,Nx:] = np.fliplr(Configuration[:Ny,:Nx])
    Configuration[Ny:,Nx:] = np.rot90(Configuration[:Ny,:Nx],2)
    
    return Configuration