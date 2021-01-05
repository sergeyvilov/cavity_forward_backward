#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 17:06:15 2020

@author: orbitalclover
"""

from pathlib import Path
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import FourierGreen as FG
import Aperture as A
import Cavity as C
import PytorchFunctions as PF


class BackwardNN(torch.nn.Module): ####just an example
    
    def __init__(self):
        super(BackwardNN, self).__init__()
        self.lin1 = torch.nn.Linear(4*Ny*Nx//4,400)
        
    def forward(self,x):
        x1 = self.lin1(x)
        return x1

##### Forward model taking Configuration0 with (bs,20x20) shape
##### and returning E field with (bs,4,80x80) shape
        
def forwardmodel(Configuration0):
    
    Configuration0 = Configuration0.view(Configuration0.shape[0],\
                                         int(np.sqrt(Configuration0.shape[1])),int(np.sqrt(Configuration0.shape[1])))
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
    
    E = torch.zeros((bs,4,Ny//2,Nx//2), device = Et['r'].device)
    E[:,0,:,:] = Et['r'][:,:Ny//2,:Nx//2,0]
    E[:,1,:,:] = Et['i'][:,:Ny//2,:Nx//2,0]
    E[:,2,:,:] = Et['r'][:,:Ny//2,:Nx//2,1]
    E[:,3,:,:] = Et['i'][:,:Ny//2,:Nx//2,1]
    
    # E = E.view(bs,4*Ny*Nx//4) #### to remove if CNN
    
    return E

class MyDataSet(Dataset):
    
    def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        x = torch.load(path_C/f'{ID}').to(device)
        y = self.labels[ID]

        return x, y
    
    
def train(model, device, training_loader, optimizer, epoch):
    
    model.train()
    
    for batch_idx, (data, target) in enumerate(training_loader):
        
        optimizer.zero_grad()
        
        output = forwardmodel(model(target))
        
        loss = F.mse_loss(output, target, reduction='sum') / output.size(0)
        
        loss.backward()
        
        optimizer.step()

        if batch_idx % 5 == 0:
            
            for batch_idx1, (data1, target1) in enumerate(validation_loader):
                                
                optimizer.zero_grad()
                
                output1 = forwardmodel(model(target1))
                
                loss_val = F.mse_loss(output1, target1, reduction='sum') / output1.size(0)

            print('Train Epoch: {:3} [{:6}/{:6} ({:3.0f}%)]\tLoss: {:.6f} Loss_val: {:.6f}'.format(
                    epoch,
                    batch_idx * len(data),
                    len(training_loader.dataset),
                    100.*batch_idx / len(training_loader,),
                    loss.item(), loss_val.item())
                )
                          
    return loss, loss_val


#### CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

#### Physical parameters
    
eps0 = 8.85418782e-12
mu0 = 1.25663706e-6
c = 1 / np.sqrt(eps0 * mu0)

L = 10e-2
H = 1.25e-2
h = 0

Npix = 20  
Nppix = 2
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

##### Propagation Operators 

Kx = np.outer(np.ones(2*Nky+1),kx)
Ky = np.outer(ky[-1::-1],np.ones(2*Nkx++1))
normKt = np.sqrt(Kx**2+Ky**2)*(np.sqrt(Kx**2+Ky**2)<k0)
normKz = np.sqrt(k0**2-normKt**2)

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

##### Aperture Operator

d = 1e-3
eps = 4.3                                                             
Z = - 1j*30/(120*np.pi)                                            
thresh = 1e-12
sgn = 1

[Tmat,Rmat] = A.DielectricLayer(eps,d,Z,kx,ky,k0,sgn,thresh)
Tmat = PF.numpytotorch(Tmat, device)
Rmat = PF.numpytotorch(Rmat, device)

RR = 15

###### Source definition

Source_field0 = np.zeros((Ny//2,Nx//2,3)) 
Source_field0[Ny//4-1:Ny//4+1,Nx//4-1:Nx//4+1,2] = 1
Source_field = PF.numpytotorch(C.symmetrizationJe(Source_field0), device)

Source_hat = PF.torchFourier(Source_field,gkyy,gxkx)
    
##### Initialization / Incident Fields
    
E_top0_hat, H_top0_hat =  PF.IncidentField(Source_hat,Gst_hat,rotGst_hat,eps0,mu0,omega)
E_bot0_hat, H_bot0_hat =  PF.IncidentField(Source_hat,Gsb_hat,rotGsb_hat,eps0,mu0,omega)

##### Definition of the dataset

path = Path('DataSet')
path_C = path/'C'
path_E = path/'E'

if os.path.exists(path) == False:
    os.mkdir(path)

if os.path.exists(path_C) == False:
    os.mkdir(path_C)
if os.path.exists(path_E) == False:
    os.mkdir(path_E)
    
bs = 100
NN = 300


for nn in range(NN):
    
    Configuration0 = (2*torch.floor(torch.rand(bs,Npix*Npix)+1/2)-1).to(device)
    E = forwardmodel(Configuration0)
    
    for ii in range(bs):
        torch.save(Configuration0[ii,:], path_C/f'{ii+nn*bs}')
        torch.save(E[ii,:], path_E/f'{ii+nn*bs}')

partition = {}

NN = 100
NN2 = 10

partition['train'] = [f'{n}' for n in range(1,NN-NN2+1)]
partition['validation'] = [f'{n}' for n in range(NN-NN2,NN+1)]

labels = {}

for n in range(1,NN+1):
    if os.path.exists(path_E/f'{n}'):
        labels[f'{n}'] = torch.load(path_E/f'{n}').to(device)
    else:
        if n < NN-NN2:
            partition['train'].remove(f'{n}')
        else:
            partition['validation'].remove(f'{n}')
            
    
training_set = MyDataSet(partition['train'], labels)
validation_set = MyDataSet(partition['validation'], labels)

bs = 10

params = {'batch_size': bs,
          'shuffle': True,
          'num_workers': 0}

training_loader = DataLoader(training_set, **params)
validation_loader = DataLoader(validation_set, **params)
 
##### Optimization

model = BackwardNN().to(device)

optimizer = torch.optim.RMSprop(model.parameters(),lr=1e-5, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False)

max_epoch = 30
          
loss_plot = torch.zeros(max_epoch)
loss_val_plot = torch.zeros(max_epoch)

for epoch in range(max_epoch):
    loss_plot[epoch], loss_val_plot[epoch] = train(model, device, training_loader, optimizer, epoch)
    
fig = plt.figure()
plt.plot(loss_plot.detach())
plt.plot(loss_val_plot.detach())
plt.show()