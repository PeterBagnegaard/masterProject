#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 13:32:26 2022

@author: peter
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import os
os.chdir('/home/peter/Desktop/master_project/cTraceo-stable')


#%%
out = loadmat('out_files/rco_python.mat')
header = out['__header__']

#%%

#import h5py
#f = h5py.File('OceanModel.mat','r')


#%%
rays = out['rays']
for ray_nbr in range(len(rays)):
    ray = rays[ray_nbr][0]
    theta = ray[0]
    r = ray[1].flatten()
    z = ray[2].flatten()
    inFile.plotInput.ax.plot(r, z, '-', color='black', linewidth = 0.5)
    plt.pause(0.2)
    
#%%
