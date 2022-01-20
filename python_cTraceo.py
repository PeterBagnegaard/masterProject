#!/usr/bin/env, python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.io import loadmat

# =============================================================================
"""                             CLASSES                                     """
# =============================================================================
""" CREATE INFILE OBJECT """
class InFile:

    def __init__(self, filename):
        self.f = open(filename + ".in", 'w')
        self.write_text(filename)
        self.write_hline()
        
    def write_hline(self):
        hline = "--------------------------------------------------------------------------------\n"
        self.f.write(hline)
        
    def write_text(self, text):
        self.f.write("'" + text + "'\n")
        
    def write_nbr(self, nbr, new_line = True):
        ending = "\n" if new_line else " "
        self.f.write(str(nbr) + ending)
        
    def write_array(self, array, new_line = True):
        for nbr in array:
            self.write_nbr(nbr, False)
        self.f.write("\n") if new_line else ""
        
    def write_arrays(self, *arrays):
        """ 
        Write arrays a0, a1, ... as 
        a0[0], a1[0], ...
        a0[1], a1[1], ...
          ⋮  ,   ⋮
        """
        n = len(arrays[0])
        for i in range(n):
            for array in arrays:
                self.write_nbr(array.flatten()[i], False)
            self.f.write("\n")
            
    def write_matrix(self, M):
        for i in range(M.shape[0]):
            h_line = M[i, :]
            self.write_array(h_line)
        
        
    def write_sourceBlock(self):
        self.write_nbr(self.sourceBlock.ds)
        self.write_array(self.sourceBlock.xx)
        self.write_array(self.sourceBlock.xbox)
        self.write_nbr(self.sourceBlock.frequency)
        """WHY DOES THIS HAVE A MINUS?!"""
        self.write_nbr(-len(self.sourceBlock.thetas)) 
        self.write_nbr(self.sourceBlock.theta_0, False)
        self.write_array(self.sourceBlock.thetas)
        self.write_hline()
        
    def write_altimetryBlock(self):
        self.write_text(self.altimetryBlock.atype)
        self.write_text(self.altimetryBlock.aptype)
        self.write_text(self.altimetryBlock.aitype)
        self.write_text(self.altimetryBlock.atiu)
        self.write_nbr(len(self.altimetryBlock.range))
        self.write_array(self.altimetryBlock.get_params())
        self.write_arrays(self.altimetryBlock.range, self.altimetryBlock.depth)
        self.write_hline()
        
    def write_soundSpeedBlock(self):
        self.write_text(self.soundSpeedBlock.cdist)
        self.write_text(self.soundSpeedBlock.cclass)
        self.write_nbr(self.soundSpeedBlock.nr0, False)
        self.write_nbr(self.soundSpeedBlock.nz0)
        if self.soundSpeedBlock.cclass != 'TABL':
            self.write_arrays(self.soundSpeedBlock.zs, self.soundSpeedBlock.cs)
        else:
            if self.soundSpeedBlock.cdist == 'c(z,z)':
                self.write_arrays(self.soundSpeedBlock.z_axis, self.soundSpeedBlock.C)
            elif self.soundSpeedBlock.cdist == 'c(r,z)':
                self.write_array(self.soundSpeedBlock.r_axis)
                self.write_array(self.soundSpeedBlock.z_axis)
                self.write_matrix(self.soundSpeedBlock.C)
        self.write_hline()
            

    def write_objectBlock(self):
        """this isn't done"""
        self.write_nbr(0)
        self.write_hline()
        
    def write_bathymetryBlock(self):
        self.write_text(self.bathymetryBlock.atype)
        self.write_text(self.bathymetryBlock.aptype)
        self.write_text(self.bathymetryBlock.aitype)
        self.write_text(self.bathymetryBlock.atiu)
        self.write_nbr(len(self.bathymetryBlock.range))
        self.write_array(self.bathymetryBlock.get_params())
        self.write_arrays(self.bathymetryBlock.range, self.bathymetryBlock.depth)
        self.write_hline()
        
    def write_arrayBlock(self):
        self.write_text(self.arrayBlock.artype)
        self.write_nbr(len(self.arrayBlock.range), False)
        self.write_nbr(len(self.arrayBlock.depth))
        self.write_array(self.arrayBlock.range)
        self.write_array(self.arrayBlock.depth)
        self.write_hline()

    def write_outputBlock(self):
        self.write_text(self.outputBlock.output)
        self.write_nbr(self.outputBlock.miss)



""" CREATE SOURCE BLOCK CLASS """
class SourceBlock:

    def __init__(self):
        pass
    
    def set_thetas(self, theta_min, theta_max, n_theta):
        self.thetas = np.linspace(theta_min, theta_max, n_theta)
        self.nthetas = len(self.thetas)


class AltimetryBlock:

    def __init__(self):
        pass
    
    def get_params(self):
        return [self.compressional_wave_speed, 
                self.shear_wave_speed, 
                self.density, 
                self.compressional_attenuation, 
                self.shear_attenuation]

    """
    Attenuation units atiu
    'F' dB/kHz
    'M' dB/meter
    'N' dB/neper
    'Q' Q factor
    'W' dB / lambda
    """
    """
    Interpolation type aitype
    'FL' flat interface
    'SL' flat interface with a slope
    '2P' piecewise linear interpolation
    '4p' piecewse cubic interpolation
    """
    """
    Interface properties atype
    'H' Homogeneous interface
    'N' Non-homogeneous interface
    """
    """
    Interface type atype
    
    'A' absorbent interface
    'E' elastic interface
    'R' rigid interface
    'V' vacuum beyond interface
    """
    """
    if aptype = ’H’
        ==FIRST LINE==
        compressional wave speed, 
        shear wave speed, 
        density, 
        compressional attenuation, 
        shear attenuation
        ==SECOND LINE==
        range(1)   , depth(1)
        ...        , ...
        range(nati), depth(nati)
    
    if aptype = "N"
        ==N'TH LINE==
        range,
        depth,
        compressional wave speed, 
        shear wave speed, 
        density, 
        compressional attenuation, 
        shear attenuation
    altimetryBlock.compressional_wave_speed = 0
    """


class SoundSpeedBlock:

    def __init__(self):
        pass
    
    def build_from_matrix(self, C, z_axis=None, r_axis=None, nr0=1):
        """ Can nr0 be whatever i want if C is one dimensional? """
        self.cclass = 'TABL'
        self.dims = self.get_nbr_dims(C)
        self.z_axis = np.arange(C.shape[0]) if z_axis is None else z_axis
        self.r_axis = np.arange(C.shape[1]) if r_axis is None else r_axis
        self.C = C
        if self.dims == 1:
            self.cdist = 'c(z,z)'
            self.nz0 = len(self.z_axis)
            self.nr0 = nr0
        elif self.dims == 2:
            self.cdist = 'c(r,z)'
            self.nz0 = len(self.z_axis)
            self.nr0 = len(self.r_axis)
        else:
            raise Exception("Wait what?! dimensions are " + str(self.dims))
            
    def get_nbr_dims(self, C):
        shape = C.shape
        if (len(shape) == 1):
            return 1
        if (len(shape) == 2):
            if (shape[0]==1 or shape[1]==1):
                return 1
            if (shape[0]>1 and shape[1]>1):
                return 2
            else:
                raise Exception("I don't even know what's wrong with " + str(shape))
        else:
            raise Exception("Wrong number of dimensions")

    """
    Type of sound speed distribution
    'c(z,z)' 
    'c(r,z)' 
    """
    """
    Class of sound speed
    'ISOV' constant
    'LINP' c0 + k(z-z0)         
    'PARP' c0 + k(z-z0)^2
    'EXPP' c0 * exp(-k(z-z0))
    'N2LP' c0 / sqrt(1+k*(z-z0))
    'ISQP' c0 * [1 + (k(z-z0)) / sqrt(1+k^2(z-z0)^2)]
    'MUNK' c0 * (1 + eps*(nabla + exp(-nabla) - 1))
    'TABL' custom sound speed profile
    if 'c(z,z)' 
        z0(1), c0(1)
        ...  , ...
        z0(nz0), c0(nz0)
    if 'c(r,z)'
        r0(1), ..., r0(nr0)
        z0(1), ..., z0(nz0)
        
        c(1, 1)  , c(1, 2), ..., c(1, nr0)
        c(2, 1)  , c(2, 2), ...,    ...
           ...   ,   ...  , ...,    ...
        c(nz0, 1),   ...  , ..., c(nz0, nr0)
    """
            

class ObjectBlock:

    def __init__(self):
        pass
    
    """
    interpolation oitype
    '2P' piecewise linear interpolation
    '4p' piecewse cubic interpolation
    """
    """
    object type otype
    'A' absorbent interface
    'E' elastic interface
    'R' rigid interface
    'V' vacuum beyond interface
    """
    """
    Attenuation units obju
    'F' dB/kHz
    'M' dB/meter
    'N' dB/neper
    'Q' Q factor
    'W' dB / lambda
    """
    """
    elastic properties compressional_wave_speed, shear_wave_speed, density, compressional_attenuation, shear_attenuation
    compressional wave speed, 
    shear wave speed, 
    density, 
    compressional attenuation, 
    shear attenuation
    """
    """
    boundaries range, down, up
    range(1)   , down(1), up(1)
    ...        , ...
    range(nati), down(nati), up(nati)
    """


class ArrayBlock:

    def __init__(self):
        pass
    
    """
    array type artype
    'RRY' Recctangular
    'HRY' Horizontal
    'VRY' Vertical
    'LRY' Linear
    """



class OutputBlock:
    
    def __init__(self):
        pass
    # Outtype
    """
    outtype output
    
    'RCO' Ray COordinates;
    'ARI' All Ray information;
    'ERF' Eigenrays (use Regula Falsi);
    ’EPR’ Eigenrays (use PRoximity method);
    ’ADR’ Amplitudes and Delays (use Regula falsi);
    ’ADP’ Amplitudes and Delays (use Proximity method);
    ’CPR’ Coherent acoustic PRessure;
    ’CTL’ Coherent Transmission Loss;
    ’PVL’ coherent Particle VeLocity;
    ’PAV’ Coherent acoustic Pressure And Particle velocity.
    """
    """
    miss
    
    distance in meters at which a ray passing a hydrophone shall be considered
    as an eigenray.
    """

"""PLOT INPUT"""
class PlotInput(InFile):
    
    def __init__(self, super):
        pass
    
    def ticks_idx(self, arr, n=10):
        res = np.round(len(arr) / n).astype(int)
        return np.round(arr[::res]).astype(int)
    
    def set_axies(self, super):
        try:
            r = super.soundSpeedBlock.r_axis
            r_idx = self.ticks_idx(r)
            self.ax.set_xticks(r_idx)
            self.ax.set_xticklabels(r_idx)
        except:
            print("No range axis to plot")
        z = super.soundSpeedBlock.z_axis
        z_idx = self.ticks_idx(z)
        self.ax.set_yticks(z_idx)
        self.ax.set_yticklabels(z_idx)
#        self.ax.set_yticks(z[z_idx])
#        self.ax.set_yticklabels(z[z_idx])
        
    def plot_input(self, super):
        """Create figure"""
        plt.close('all')
        self.fig, self.ax = plt.subplots(1, 1)
        
        """sound speed"""
        if (super.soundSpeedBlock.cclass == 'TABL'):
            r_min, r_max = super.soundSpeedBlock.r_axis.min(), super.soundSpeedBlock.r_axis.max(); r_padding = (r_max - r_min) * 0.01
            z_min, z_max = super.soundSpeedBlock.z_axis.min(), super.soundSpeedBlock.z_axis.max(); z_padding = (z_max - z_min) * 0.01
            self.ax.imshow(super.soundSpeedBlock.C, cmap='Blues', extent=[r_min, r_max, z_max, z_min])
#            self.ax.imshow(super.soundSpeedBlock.C, aspect='auto', cmap='Blues', extent=[r_min, r_max, z_max, z_min])
#            self.ax.imshow(super.soundSpeedBlock.C, aspect='auto', cmap='Blues', extent=[r_min, r_max, z_min, z_max])
            plt.plot([r_min - r_padding, r_max + r_padding], [z_min - z_padding, z_max + z_padding], linewidth = 0)
            self.set_axies(super)
        
        """Source"""
        self.ax.plot(super.sourceBlock.xx[0], super.sourceBlock.xx[1], '*r', label="Source", markersize = 15)
        
        """altimetry"""
        self.ax.plot(super.altimetryBlock.range, super.altimetryBlock.depth, '-c', label="altimetry", linewidth = 5)
        
        """bathymetry"""
        self.ax.plot(super.bathymetryBlock.range, super.bathymetryBlock.depth, '-m', label="bathymetry", linewidth = 5)
        
        """array"""
        for r_i in super.arrayBlock.range:
            for d_i in super.arrayBlock.depth:
                self.ax.plot(r_i, d_i, '*g', markersize = 15)
        self.ax.plot(r_i, d_i, '*g', label='Reciever', markersize = 15)
        self.ax.legend()


if __name__ == '__main__':
    os.chdir('/home/peter/Desktop/master_project/cTraceo-stable/out_files')
    
#%% soource 
    sourceBlock = SourceBlock()
    # ray step [m]
    sourceBlock.ds = 1.
    # Source coordinates [m]
    sourceBlock.xx = [0., 25.]
    # Range box [m]
    sourceBlock.xbox = [-1., 200.]
    #sourceBlock.xbox = [-1., 1000.]
    # Source frequency [Hz]
    sourceBlock.frequency = 100.
    # Number of launching angles [degrees]
    sourceBlock.theta_0 = 0.6
    sourceBlock.set_thetas(-60., 30., 101)
    
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
    altimetryBlock.compressional_wave_speed = 1510.
    altimetryBlock.shear_wave_speed = 300.
    altimetryBlock.density = 2.
    altimetryBlock.compressional_attenuation = 0.
    altimetryBlock.shear_attenuation = 0.
    # coordinates
    altimetryBlock.range = np.array([-2., 202.])
    altimetryBlock.depth = np.array([0., 0.])
    
    #%% Sound Speed
    soundSpeedBlock = SoundSpeedBlock()
    oceanmodel = loadmat('OceanModel.mat')['OceanModel'][::3, ::10]
    soundSpeedBlock.build_from_matrix(oceanmodel)
    # Type of sound speed distribution
    #soundSpeedBlock.cdist = 'c(z,z)'
    # Class of sound speed
    #soundSpeedBlock.cclass = 'ISOV'
    # nbr points in range, nbr of points in depts
    #soundSpeedBlock.nr0 = 10
    #soundSpeedBlock.nz0 = 20
    # parameters of profiles
    #soundSpeedBlock.zs = np.array([0., 100.])
    #soundSpeedBlock.cs = np.array([1500., 1500.])

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
    bathymetryBlock.atype = 'R'
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
    bathymetryBlock.shear_wave_speed = 0.
    bathymetryBlock.density = 2.
    bathymetryBlock.compressional_attenuation = 0.
    bathymetryBlock.shear_attenuation = 0.
    # coordinates
    bathymetryBlock.range = np.array([-2., 202.])
    bathymetryBlock.depth = np.array([160., 160.])
    
    #%% array
    arrayBlock = ArrayBlock()
    # array type
    arrayBlock.artype = 'HRY'
    arrayBlock.range = np.linspace(0., 199., 16)
    arrayBlock.depth = np.array([150.])
    #arrayBlock.nra = len(arrayBlock.range)
    #arrayBlock.nza = len(arrayBlock.depth)

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
    title="my_title"
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
    print("file " + title +".in created")


    os.system('ctraceo ' + title)
    print("cTraceo " + title + " run")