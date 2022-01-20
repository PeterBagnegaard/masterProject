#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 11:10:39 2022

@author: peter
"""
import sys; sys.path.append('/home/peter/Desktop/master_project/cTraceo-stable')
import os; os.chdir('/home/peter/Desktop/master_project/cTraceo-stable/out_files')
#from python_cTraceo import *
from python_cTraceo import SourceBlock, AltimetryBlock, ObjectBlock, ArrayBlock, SoundSpeedBlock, OutputBlock, InFile, PlotInput
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

def blur(image):
    return (np.roll(image, 1, axis=1) + np.roll(image, 1, axis=-1) + np.roll(image, -1, axis=1) + np.roll(image, -1, axis=-1) + image) / 5

def blur_n(image, n=5):
    for _ in range(5):
        image = blur(image)
    return image

def munk(z, c0, z0):
    eps = 7.4 * 10**-3
    B = 1.3
    eta = 2 * (z - z0) / B
    c = c0 * (1 + eps * (eta + np.exp(-eta) - 1))
    return c.reshape([-1, 1])


#%% soource 
sourceBlock = SourceBlock()
# ray step [m]
sourceBlock.ds = 11.
# Source coordinates [m]
sourceBlock.xx = [1000., 1000.]
# Range box [m]
sourceBlock.xbox = [-1., 101000.]
sourceBlock.xbox = [-1., 5010.]
# Source frequency [Hz]
sourceBlock.frequency = 50.
# Number of launching angles [degrees]
sourceBlock.theta_0 = 0.56
sourceBlock.set_thetas(-20., -14., 51)

#%% altimetry
altimetryBlock = AltimetryBlock()
# Interface type
altimetryBlock.atype = 'V'        
# Interface properties
altimetryBlock.aptype = 'H'
# Interpolation type
altimetryBlock.aitype = 'FL'
# Attenuation units
altimetryBlock.atiu = "W"
# number of interface coordinates
altimetryBlock.nati = 2
# interface properties
altimetryBlock.compressional_wave_speed = 0.#1510.
altimetryBlock.shear_wave_speed = 0.#300.
altimetryBlock.density = 0.#2.
altimetryBlock.compressional_attenuation = 0.
altimetryBlock.shear_attenuation = 0.
# coordinates
altimetryBlock.range = np.array([-2., 101002.])
altimetryBlock.range = np.array([-2., 5012.])
altimetryBlock.depth = np.array([500., 500.])


#%% Objects THIS IS NOT DONE !!!
objectBlock = ObjectBlock()
# number of objects
#objectBlock.nobj = 0 CALCULATE THIS
# interpolation
objectBlock.oitype = '2p'
# object type
objectBlock.otype = 'A'
# Attenuation units
objectBlock.obju = "W"
# number of coordinates
#objectBlock.no = 4 CALCULATE THIS
# elastic properties
objectBlock.compressional_wave_speed = 1510.
objectBlock.shear_wave_speed = 300.
objectBlock.density = 2.
objectBlock.compressional_attenuation = 0.
objectBlock.shear_attenuation = 0.
# boundaries
objectBlock.range = np.array([1., 2., 3.])
objectBlock.down = np.array([50., 51., 50.])
objectBlock.up = np.array([50., 49., 50.])

#%% bathymetry
bathymetryBlock = AltimetryBlock()
# Interface type
bathymetryBlock.atype = 'E'
# Interface properties
bathymetryBlock.aptype = 'H'
# Interpolation type
bathymetryBlock.aitype = 'FL'
# Attenuation units
bathymetryBlock.atiu = "W"
# number of interface coordinates
bathymetryBlock.nati = 2
# interface properties
bathymetryBlock.compressional_wave_speed = 1550.
bathymetryBlock.shear_wave_speed = 600.
bathymetryBlock.density = 2.
bathymetryBlock.compressional_attenuation = 0.1
bathymetryBlock.shear_attenuation = 0.
# coordinates
bathymetryBlock.range = np.array([-2., 101002.])
bathymetryBlock.range = np.array([-2., 5012.])
bathymetryBlock.depth = np.array([5000., 5000.])

#%% array
arrayBlock = ArrayBlock()
# array type
arrayBlock.artype = 'RRY'
arrayBlock.range = np.array([101000.])
arrayBlock.range = np.array([5010.])
arrayBlock.depth = np.array([5000.])
#arrayBlock.range = np.linspace(0., 199., 16)
#arrayBlock.depth = np.array([150.])

#%% Sound Speed
oceanmodel = loadmat('OceanModel.mat')['OceanModel']#[::2, ::3]
oceanmodel = blur_n(oceanmodel, n=150)

r_min = min(bathymetryBlock.range.min(), altimetryBlock.range.min())
r_max = max(bathymetryBlock.range.max(), altimetryBlock.range.max())
z_min = min(bathymetryBlock.depth.min(), altimetryBlock.depth.min())
z_max = max(bathymetryBlock.depth.max(), altimetryBlock.depth.max())
z_ax = np.linspace(z_min, z_max,  oceanmodel.shape[0])
r_ax = np.linspace(r_min, r_max,  oceanmodel.shape[1])

soundSpeedBlock = SoundSpeedBlock()

""" USE 1D MODEL """
z = np.linspace(0., 5000., 1001)
c = munk(z_ax/1000, 1500, 1.300)
soundSpeedBlock.build_from_matrix(c, z_ax, r_ax)
""" USE 2D MODEL """
#soundSpeedBlock.build_from_matrix(oceanmodel, z_ax, r_ax)

""" USE BUILD-IN METHOD """
##soundSpeedBlock.build_from_matrix(oceanmodel)
## Type of sound speed distribution
#soundSpeedBlock.cdist = 'c(z,z)'
## Class of sound speed
#soundSpeedBlock.cclass = 'TABL'
## nbr points in range, nbr of points in depts
#soundSpeedBlock.nr0 = 1
#soundSpeedBlock.nz0 = 1001
## parameters of profiles
#soundSpeedBlock.zs = np.array([0., 100.])
#soundSpeedBlock.cs = np.array([1500., 1500.])

#%% output
outputBlock = OutputBlock()
# Output type
outputBlock.output = 'RCO'
# Miss (used for eigen rays)
outputBlock.miss = 0.5

#%%
"""
Create .in file
"""        
title = "my_title"
out_title = "_python"
inFile = InFile(title)

inFile.sourceBlock = sourceBlock
inFile.altimetryBlock = altimetryBlock
inFile.soundSpeedBlock = soundSpeedBlock
inFile.objectBlock = objectBlock
inFile.bathymetryBlock = bathymetryBlock
inFile.arrayBlock = arrayBlock
inFile.outputBlock = outputBlock
inFile.plotInput = PlotInput(inFile)

inFile.plotInput.plot_input(inFile)
inFile.write_sourceBlock()
inFile.write_altimetryBlock()
inFile.write_soundSpeedBlock()
inFile.write_objectBlock()
inFile.write_bathymetryBlock()
inFile.write_arrayBlock()
inFile.write_outputBlock()

inFile.f.close()
print(title + ".in created")

os.system('ctraceo ' + title)

standard_out_name = inFile.outputBlock.output.lower()
os.rename(standard_out_name + ".mat", standard_out_name + out_title + ".mat")

print(standard_out_name + out_title + ".mat created")
#print('ctraceo ' + title + " => " + standard_out_name + out_title + ".mat")
