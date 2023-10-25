# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 23:37:58 2022

@author: samith
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import gc
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt
import itertools

import pandas as pd
from tqdm import tqdm
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad



class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
    
    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords        

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        #activations = OrderedDict()

        # activation_count = 0
        # x = coords.clone().detach().requires_grad_(True)
        # activations['input'] = x
        # for i, layer in enumerate(self.net):
        #     if isinstance(layer, SineLayer):
        #         x, intermed = layer.forward_with_intermediate(x)
                
        #         if retain_grad:
        #             x.retain_grad()
        #             intermed.retain_grad()
                    
        #         activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
        #         activation_count += 1
        #     else: 
        #         x = layer(x)
                
        #         if retain_grad:
        #             x.retain_grad()
                    
        #     activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
        #     activation_count += 1

        # return activations
    
class IMUDATA(torch.utils.data.Dataset):
    def __init__(self, filename,f):
        df = pd.read_csv(filename)
        timeLimL = 1000
        timeLimU = 2000
        columX = 1 #added remove erros
        self.sequence = df.iloc[timeLimL:timeLimU,0].values
        self.sequence = np.array([self.sequence[i] for i in range(0,self.sequence.shape[0],f)])
        self.data = df.iloc[timeLimL:timeLimU,columX].values
        self.data = np.array([self.data[i] for i in range(0,self.data.shape[0],f)])
        self.data = self.data.astype(np.float32)
        self.sequence = self.sequence.astype(np.float32)
        self.timepoints = get_mgrid(len(self.data), 1)

    def get_num_samples(self):
        return self.sequence.shape[0]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        
        amplitude = self.data
        scale = np.max(np.abs(amplitude))
        amplitude = (amplitude / scale)
        amplitude = torch.Tensor(amplitude).view(-1,1)
      
        return self.timepoints, amplitude
    

def train_Siren(data_sequence,train_steps,sample_freq,siren_omega):
    
    #define the training steps
    total_steps = train_steps 
    steps_til_summary = 100
    sample_freq=sample_freq
    
    
    #time points to sample at high freq
    timepoints_highf = get_mgrid(len(data_sequence), 1).to(device)
    
    #sample the data in sample_freq
    dataloaderSampled=np.array([data_sequence[i] for i in range(0,data_sequence.shape[0],sample_freq)])
    # print(len(dataloaderSampled))
    timepoints_lowf = get_mgrid(len(dataloaderSampled), 1)
    
    #generate model parameters
    
    in_features = 1
    out_features = data_sequence.shape[1]
    
    audio_siren = Siren(in_features=1, out_features=out_features, hidden_features=256, 
                        hidden_layers=3, first_omega_0=siren_omega, outermost_linear=True)
    audio_siren.to(device)
    
    optim = torch.optim.Adam(lr=1e-4, params=audio_siren.parameters())
    model_input, ground_truth = timepoints_lowf, torch.Tensor(dataloaderSampled)
    # torch.from_numpy
    model_input = model_input.to(device)
    ground_truth=ground_truth.to(device)
    
    
    # model_inputN, ground_truthN = next(iter(dataloader))
    # model_inputN, ground_truthN = model_inputN.cuda(), ground_truthN.cuda()
   
    for step in tqdm(range(total_steps),desc='Progress'):
        model_output, coords = audio_siren(model_input)    
        loss = F.mse_loss(model_output, ground_truth)
        
        # if not step % steps_til_summary:
        #     print("Step %d, Total loss %0.6f" % (step, loss))
        
        #     fig, axes = plt.subplots(1,2,figsize=(18,7))
        #     axes[0].plot(coords.squeeze().detach().cpu().numpy(),model_output.squeeze().detach().cpu().numpy())
        #     axes[1].plot(coords.squeeze().detach().cpu().numpy(),ground_truth.squeeze().detach().cpu().numpy())
        #     plt.show() 
        optim.zero_grad()
        loss.backward()
        optim.step()
    # del model_output,coords,model_input,ground_truth,loss

    final_model_output, coords = audio_siren(timepoints_highf)
    generated_targets = final_model_output.cpu().detach().squeeze().numpy()
    #print('generated data lenght: ',generated_targets.shape[0], 'original lenght: ', len(data_sequence))
    del audio_siren,final_model_output, coords
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("Trajectory lenght: " , len(data_sequence),len(generated_targets))
    return generated_targets



# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18,7))

# error = (ground_truthN.cpu().detach().squeeze().numpy() - final_model_output.cpu().detach().squeeze().numpy())

# axes[0].plot(coords.squeeze().detach().cpu().numpy(),final_model_output.squeeze().detach().cpu().numpy())
# axes[1].plot(coords.squeeze().detach().cpu().numpy(),ground_truthN.squeeze().detach().cpu().numpy())
# axes[2].plot(coords.squeeze().detach().cpu().numpy(),error)

# #plt.plot(sequence, z, label="line z")

# plt.show()

def genrate_gtpose():
    print('gt pose generated')
    

                
                
                
                
                
                
                