#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 14:15:01 2022

@author: peter
"""
from time import perf_counter
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
import os
from lib import (numpy2rsf, 
                 rsf2numpy, 
                 create_hist, 
                 extremum, 
                 fix_comma_problem, 
                 ThermoclineSpline,
                 plot_source_receiver, 
                 plot_time, 
                 SeabedSpline,
                 write_file,
                 OceanModel,
                 Inversion,
                 log
                 )

def compare_ocean_models(true_oceanModel, test_oceanModel):
    true_thermo = true_oceanModel.get_thermocline_state()
    test_thermo = test_oceanModel.get_thermocline_state()
    print("Difference between thermocline parameters")
    print(true_thermo - test_thermo)
    
    true_seabed = true_oceanModel.seabed_spline.seabed_coordinates()[0]
    test_seabed = test_oceanModel.seabed_spline.seabed_coordinates()[0]
    print("Difference between seabed points")
    print(true_seabed - test_seabed)
    
    true_times = true_oceanModel.times
    test_times = test_oceanModel.times
    print("Difference between test times")
    print(true_times - test_times)
    
    plt.figure()
    plt.hist(true_oceanModel.TOA() / test_oceanModel.TOA(), bins=20)

def save_inversion_figures(inversion):
    figa = inversion.plot_switching_criteria()
    manager = plt.get_current_fig_manager(); manager.full_screen_toggle()
    
    plt.pause(2)
    figa.savefig('Switching_criteria.png', bbox_inches='tight',pad_inches = 0)

def save_oceanmodel_figures(oceanModel, n_idx=0, r_idx=9):
    figa = oceanModel.plot_oceanmodel()
    manager = plt.get_current_fig_manager(); manager.full_screen_toggle()
    figb = oceanModel.plot_source_receiver([n_idx], [r_idx])
    manager = plt.get_current_fig_manager(); manager.full_screen_toggle()
    figc = oceanModel.plot_time([n_idx], [r_idx])
    manager = plt.get_current_fig_manager(); manager.full_screen_toggle()
    figd = oceanModel.plot_travel_routes([n_idx], [r_idx])
    manager = plt.get_current_fig_manager(); manager.full_screen_toggle()
    fige = oceanModel.plot_TOA()
    manager = plt.get_current_fig_manager(); manager.full_screen_toggle()
    
    plt.pause(2)
    figa.savefig('OceanModel.png', bbox_inches='tight',pad_inches = 0)
    figb.savefig('Eikonal.png', bbox_inches='tight',pad_inches = 0)
    figc.savefig('Travel_Time.png', bbox_inches='tight',pad_inches = 0)
    figd.savefig('Travel_Route.png', bbox_inches='tight',pad_inches = 0)
    fige.savefig('TOA.png', bbox_inches='tight',pad_inches = 0)

def clean_rsf():
    all_files = os.listdir()
    rsf_files = [file for file in all_files if file.endswith('.rsf')]
    for rsf_file in rsf_files:
        os.system(f'sfrm {rsf_file}')

def clean_rsf_deep():
    cur_dir = os.getcwd()
    os.chdir('/var/tmp')
    all_files = os.listdir()
    rsf_files = [file for file in all_files if file.endswith('.rsf@')]
    for rsf_file in rsf_files:
        os.system(f'rm {rsf_file}')
    os.chdir(cur_dir)
    return rsf_files    

def get_sigma_matrix(oceanModel, sigma=10**-6):
    ns = len(oceanModel.sources)
    nr = len(oceanModel.receivers)
    return np.ones(ns * nr) * sigma

def get_true_data():
    """
    Load OceanModel.mat to np array
    """
    _true_oceanmodel = loadmat('OceanModel.mat')['OceanModel'].astype(np.float32)
    _true_oceanmodel = gaussian_filter(_true_oceanmodel, 5)
    return _true_oceanmodel

def get_uniform_field(_true_oceanmodel, speed=1500.):
    """
    Get test data that looks like true data
    """
    if isinstance(_true_oceanmodel, list) and len(_true_oceanmodel) == 2:
        om = np.ones(_true_oceanmodel) * speed
    elif isinstance(_true_oceanmodel, np.ndarray):
        om = np.ones_like(_true_oceanmodel) * speed
    else:
        raise Exception(f"get_uniform_field got wrong input of type {type(_true_oceanmodel)}")
    return om.astype(np.float32)

def get_slope(oceanmodel, a, b):
    oceanmodel = np.ones_like(oceanmodel) * 1500.
    v, h = oceanmodel.shape
    A = v / h * a
    B = b * v
    
    x_ax = np.arange(h)
    y_ax = A * x_ax + B
    for x, y in zip(x_ax, y_ax):
        oceanmodel[int(y):, int(x)] = 4000.
    return oceanmodel

def flat_seabed(n_spline, h_span, seabed_height, noise_fraction=0.):
    h_ax = np.linspace(0, h_span, num=n_spline) #
    v_ax = np.ones_like(h_ax) * seabed_height
    noise = v_ax * np.random.randn(n_spline) * noise_fraction
    return (h_ax, v_ax + noise)

def set_true_seabed(inversion, true_oceanModel, scrample=False):
    points = true_oceanModel.seabed_spline.seabed_coordinates(True)[1]
    if scrample:
        points[5] = 1.1*points[5]
    inversion.set_seabed_spline(points)

plt.close('all')
os.chdir('/home/peter/Desktop/master_project/Madagascar/MadClass')

# Set what to log
log.log_function_calls_mode = False
log.log_function_counts_mode = True

"""
Create OceanModel Classes
coordinates [horizontal, vertical]
"""
_oceanmodel = get_uniform_field([500, 1985])

sources   = [[i, 0.1] for i in np.linspace(0.1, 1.3, num=10)]
receivers = [[i, 0.1] for i in np.linspace(0.1, 1.3, num=10)+0.05]
times = np.linspace(0, 20, num=len(sources))

h0, v0 = flat_seabed(10, _oceanmodel.shape[1], 400, noise_fraction=0. )
h1, v1 = flat_seabed(10, _oceanmodel.shape[1], 400, noise_fraction=0.1)
#h1, v1 = np.copy(h0), np.array([400., 400., 400., 380., 400., 410., 370., 400., 400., 400.])

print("Creating ocean models")

true_oceanModel = OceanModel(_oceanmodel, sources, receivers, times, os.getcwd(), hv_points=(h1, v1), step_sizes=[0.0007, 0.0007, 0.0005], thermo_depth=250., thermo_amplitude=15., thermo_wave_length=500., verbose=False)
test_oceanModel = OceanModel(_oceanmodel, sources, receivers, times, os.getcwd(), hv_points=(h0, v0), step_sizes=[0.0007, 0.0007, 0.0005], thermo_depth=250., thermo_amplitude=10., thermo_wave_length=400., verbose=False)

true_oceanModel.initialize()

cp_1 = log.checkpoint()
print("Ocean models created")

"""
Save oceanmodel figures 
"""
#save_oceanmodel_figures(true_oceanModel)

#%%
"""
Initialize Inversion class
"""
inversion = Inversion(test_oceanModel, sigmas=get_sigma_matrix(true_oceanModel), verbose=True)

inversion.set_true_data(true_oceanModel)
inversion.know_the_real_answer(true_oceanModel.seabed_spline, true_oceanModel.get_thermocline_state())

set_true_seabed(inversion, true_oceanModel)

cp_2 = log.checkpoint()

#%%
t_0 = perf_counter()
cost, errors, fig, a_ax, w_ax = inversion.plot_cost_around_thermocline(true_oceanModel, a_diff=20, w_diff=200, a_num=2, w_num=2)#, a_ax_=np.linspace(0, 100, num=100))
cp_3 = log.checkpoint()
dt = perf_counter() - t_0

print(dt)
print(f"{len(a_ax)}, {len(w_ax)} = {len(a_ax) * len(w_ax)}")
print(f"{len(a_ax) * len(w_ax)} iterations of {dt / (len(a_ax) * len(w_ax)):.3}s took {dt/60:.3}m in total")

#%%

# dt = 217
# a * w = 5 * 5 = 25
# dt / (a * w) = 8.68

# dt = 67
# a * w = 3 * 3 = 9
# dt / (a * w) = 7.48

# dt = 137
# a * w = 5 * 3 = 15
# dt / (a * w) = 9.17

# dt = 286
# a * w = 7 * 7 = 49
# dt / (a * w) = 5.83

# dt = 141.84794797599898
# 5, 5 = 25
# 5.67

# 215.43505610099965
# 5, 7 = 35
# 6.15
N = [25, 9, 15, 49, 25, 35]
T = [8.682780869840062, 7.488783304332982, 9.174675619066694, 5.834121903918366, 5.673917919039959, 6.155287317171418]

plt.figure()
plt.plot(N[:3], T[:3], '.r')
plt.plot(N[3:], T[3:], '.g')

# sec pr hour T
hr = 60*60

# sec pr pixel t / p
spp = max(T[3:])

# pixels in an hour In an hour T / (t / p) = T * (p / t) = T / t * p
print(hr / spp)

# n * n
print(np.sqrt(hr / spp))

#%%
print("Beginning inversion!")
"""
Solve seabed inversely
"""
correction = 1.#0.5*10**4
best_c, best_model, best_idx, i = inversion.Solve(dv=1., dthermo=1., alpha_seabed=correction*4*10**6, alpha_thermo=correction*10**4, max_iter=50, min_iter=0, thermocline_iter=60, plot_optimization=True, only_optimize_thermocline=True)

cp_3 = log.checkpoint()
print("Ending inversion!")

#%%
#plt.close('all')

data = np.array(inversion.data)

a = data[:, 1]
da = data[:, 0]
#da = a[1:] - a[:-1]; a = a[1:]
w = data[:, 3]
dw = data[:, 2]
#dw = w[1:] - w[:-1]; w = w[1:]
t = data[:, 5]
dt = data[:, 4]
dt = dt[t!=0.]
t = t[t!=0.]
#dt = t[1:] - t[:-1]; t = t[1:]


def plot_thermo_hist(a, da, w, dw,  t, dt):
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.hist(da / a, bins=30)
    plt.ylabel("Amplitude")
    plt.subplot(3, 1, 2)
    plt.hist(dw / w, bins=30)
    plt.ylabel("Wave length")
    plt.subplot(3, 1, 3)
    plt.hist(dt / t, bins=30)
    plt.ylabel("Time")

def plot_thermo_vs_iteration(a, da, w, dw,  t, dt):
    plt.figure()
    plt.subplot(3, 3, 1)
    plt.plot(a, '.')
    plt.ylabel("Amplitude")
    plt.subplot(3, 3, 2)
    plt.plot(da, '.')
    plt.subplot(3, 3, 3)
    plt.plot(da / a, '.')
    
    plt.subplot(3, 3, 4)
    plt.plot(w, '.')
    plt.ylabel("Wave length")
    plt.subplot(3, 3, 5)
    plt.plot(dw, '.')
    plt.subplot(3, 3, 6)
    plt.plot(dw / w, '.')
    
    plt.subplot(3, 3, 7)
    plt.plot(t, '.')
    plt.ylabel("Time")
    plt.subplot(3, 3, 8)
    plt.plot(dt, '.')
    plt.subplot(3, 3, 9)
    plt.plot(dt / t, '.')

plot_thermo_hist(a, da, w, dw,  t, dt)
plot_thermo_vs_iteration(a, da, w, dw,  t, dt)


#%%
#def cleanup_ratios(ratios, fill_val=None):
#    ratios = np.array(ratios)
#    _fill_val = ratios.max() if fill_val is None else fill_val
#    nan = np.isnan(ratios)
#    return 

ratios = np.array(inversion.ratio)
idx = np.isinf(ratios)
idx_not = ~idx
ratios[idx] = ratios[idx_not].max()
ratios[idx] = 0#ratios[idx_not].max()
        
for i in range(ratios.shape[1]):
    plt.subplot(2, 3, i+1)
    plt.hist(ratios[:, i], bins=20)

for i in range(ratios.shape[1]):
    plt.subplot(2, 3, i+1+ratios.shape[1])
    plt.plot(ratios[:, i], '.')

#%%
a = np.array(inversion.a_list)
w = np.array(inversion.w_list)
t = np.array(inversion.t_list)

plt.ylabel("a")
plt.subplot(3, 3, 1)
plt.plot(a[:, 0], '.')
plt.subplot(3, 3, 2)
plt.plot(a[:, 1], '.')
plt.subplot(3, 3, 3)
plt.plot(a[:, 2], '.')
plt.ylabel("w")
plt.subplot(3, 3, 4)
plt.plot(w[:, 0], '.')
plt.subplot(3, 3, 5)
plt.plot(w[:, 1], '.')
plt.subplot(3, 3, 6)
plt.plot(w[:, 2], '.')
plt.ylabel("t")
plt.subplot(3, 3, 7)
plt.plot(t[:, 0], '.')
plt.subplot(3, 3, 8)
plt.plot(t[:, 1], '.')
plt.subplot(3, 3, 9)
plt.plot(t[:, 2], '.')

#%%
"""
Save oceanmodel figures 
"""
save_inversion_figures(inversion)

#%%
"""
Plot iteration history of each seabed point
"""
import matplotlib.gridspec as gridspec
plt.close('all')
linewidth = 5
label_fontsize = 15
title_fontsize = 20
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

gs = gridspec.GridSpec(nrows=3, ncols=1)
fig = plt.figure()
ax0 = fig.add_subplot(gs[0:2, 0])
ax1 = fig.add_subplot(gs[2, 0], sharex=ax0)
fig.tight_layout()
fig.subplots_adjust(hspace=0)   

seabeds = np.array(inversion.inversion_history)[:, :10]
iter_ax = np.arange(seabeds.shape[0])

for seabed, color in zip(seabeds.T, colors):
    ax0.plot(iter_ax, seabed, ':', color=color, linewidth=linewidth)    
for test, true, color in zip(seabeds[-1, :], inversion.true_seabed, colors):
    ax0.plot([iter_ax[-1], iter_ax[-1]+1], [test, true], '-', color=color, linewidth=linewidth)
ax0.set_ylabel("Depth of seabed  point", fontsize=label_fontsize)

ax1.plot(iter_ax, inversion.cs, ':k', color='black', linewidth=linewidth)
ax1.set_xlabel("Iteration", fontsize=label_fontsize)
ax1.set_ylabel("Cost", fontsize=label_fontsize)

ax0.set_title("Iteration history of seabed points", fontsize=title_fontsize)

#%%
"""
Plotting inversion history
"""
#label_fontsize = 15
#title_fontsize = 20
#markersize = 15
#linewidth = 5
#
#
#if len(inversion.cs) == 0:
#    raise Exception("No inversion history found!")
#    
#fig, (ax0, ax1) = plt.subplots(1, 2)
#fig.tight_layout()
#
#h_ax = inversion.s_horizontal
#best_idx = np.argmin(inversion.cs)
#for i, seabed_spline in enumerate(inversion.inversion_history[:inversion.switch_idx]):
#    alpha = 0.5 * i / len(inversion.inversion_history)
#    ax0.plot(h_ax, seabed_spline[:len(h_ax)], '-k', alpha=alpha, linewidth=linewidth)
#    plt.pause(0.1)
#    
#if inversion.true_seabed is not None:
#    ax0.plot(h_ax, inversion.true_seabed, '-r', linewidth=linewidth, label="True model")
#ax0.plot(h_ax, inversion.inversion_history[0][:len(h_ax)], '-k', linewidth=linewidth, label="Initial model")
#ax0.plot(h_ax, inversion.inversion_history[best_idx][:len(h_ax)], '-b', linewidth=linewidth, label="Recovered model")
#
#ax0.set_xlabel("Horizontal distance", fontsize=label_fontsize)
#ax0.set_ylabel("Depth", fontsize=label_fontsize)
#ax1.plot(inversion.cs, '*:r', label="Cost", linewidth=linewidth)
#ax1.text(30, 4*10**-5, f"Smallest cost: \n{inversion.cs[best_idx]:.2}", fontsize=label_fontsize)
#ax0.legend(fontsize=label_fontsize)
#ax1.set_xlabel("Iterations", fontsize=label_fontsize)
#ax1.set_ylabel("Cost", fontsize=label_fontsize)
#ax1.legend(fontsize=label_fontsize)
#fig.suptitle("Plot of inversion method", fontsize=title_fontsize)

#%%
"""
Plot how switching criteria works
"""
#def diff1d(s):
#    ds = np.roll(s, -1) - s
#    ds[-1] = ds[-2]
#    return ds
#
#def plot_switching_criteria():
#    
#    def get_S(cs):
#        return np.array([np.std(cs[:i+1]) for i in range(len(cs))])
#    
#    def get_dS(cs):
#        S = get_S(cs)
#        return diff1d(S)
#    
#    linewidth = 5
#    label_fontsize = 15
#    title_fontsize = 20
#    
#    fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
#    fig.tight_layout()
#    fig.subplots_adjust(hspace=0)   
#    
#    cs = np.array(inversion.cs)
#    S = get_S(cs)
#    dS = get_dS(cs)
#    iter_ax = np.arange(len(cs))
#    ok = dS > 0
#    switch = np.argmin(ok[5:]) + 4
#    
#    ax0.plot(iter_ax, cs, '-r', linewidth=linewidth)
#    ax0.plot(iter_ax[ok], cs[ok], '-g', linewidth=linewidth)
#    ax0.set_ylabel("Cost: c(i)",  fontsize=label_fontsize)
#    ax0.axvline(2, linewidth=linewidth/2, color='black', ls=':', label='Initial skipped steps')
#    ax0.axvline(switch, linewidth=linewidth/2, color='black', label='Point of switch')
#    ax0.set_xlim([0, iter_ax.max()])
#    ax1.plot(iter_ax, S, '-r', linewidth=linewidth)
#    ax1.plot(iter_ax[ok], S[ok], '-g', linewidth=linewidth)
#    ax1.set_ylabel("S = std(c(:i))",  fontsize=label_fontsize)
#    ax1.axvline(2, linewidth=linewidth/2, color='black', ls=':')
#    ax1.axvline(switch, linewidth=linewidth/2, color='black')
#    ax1.set_xlim([0, iter_ax.max()])
#    ax2.plot(iter_ax, dS, '-r', linewidth=linewidth)
#    ax2.plot(iter_ax[ok], dS[ok], '-g', linewidth=linewidth)
#    ax2.set_ylabel(r"$\frac{dS}{di}$",  fontsize=title_fontsize)
#    ax2.set_xlabel("Iterations (i)", fontsize=label_fontsize)
#    ax2.axvline(2, linewidth=linewidth/2, color='black', ls=':')
#    ax2.axvline(switch, linewidth=linewidth/2, color='black')
#    ax2.axhline(0, linewidth=linewidth/2, color='black', ls='--')
#    ax2.set_xlim([0, iter_ax.max()])
#    ax0.set_title("Determining switching criteria during optimization", fontsize=title_fontsize)
#    fig.legend(fontsize=label_fontsize)
#    
#    return fig
#
#plot_switching_criteria()
#%%
"""
Print names of unised functions
"""
unused_functions = []
used_functions = list(cp_1.function_counts.keys()) + list(cp_2.function_counts.keys()) + list(cp_3.function_counts.keys())
for func_name in log.functions:
    if not func_name in used_functions:
        unused_functions.append(func_name)
print(unused_functions)


