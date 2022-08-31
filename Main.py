#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 14:15:01 2022

@author: peter
"""
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

def plot_cost_field_1d(ax, cost, axis, axis_0):
    ax.plot(axis, cost, '.-r', markersize=5)
    ax.set_ylabel("cost")
    ax.set_xlabel("Amplitude")
    ax.axvline(axis_0)
    
def plot_cost_field(cost, a_ax, w_ax, a_0, w_0, fig=None):
    if fig is None:
        fig, ax = plt.subplots(1, 1)
    fig.clear()
    fig.add_subplot()
    ax = fig.get_axes()[0]
    j = cost[cost != 0]
    im = ax.imshow(cost, extent=[w_ax[0], w_ax[-1], a_ax[-1], a_ax[0]], aspect='auto', vmin=j.min(), vmax=j.max())
    ax.set_xlabel('Wave number')
    ax.set_ylabel('Amplitude')
    ax.plot([w_0], [a_0], '*r')
    plt.colorbar(im)
    plt.pause(0.01)
    return fig

def plot_cost_around_thermocline(inversion, true_oceanModel, a_diff=3, w_diff=100, t_diff=100, a_num=12, w_num=12, t_num=12, a_ax_=None, w_ax_=None, t_ax_=None):
    
    def get_parameter_ax(p_0, diff, num):
        half = int(num/2)
        a = np.linspace(-diff, 0, num=half)
        b = np.linspace(0, diff, num=num-half+1)
        return np.concatenate((a, b[1:])) + p_0

    errors = []
    
    a_0, w_0, t_0 = true_oceanModel.get_thermocline_state()
    
    a_ax = get_parameter_ax(a_0, a_diff, a_num) if a_ax_ is None else a_ax_
    w_ax = get_parameter_ax(w_0, w_diff, w_num) if w_ax_ is None else w_ax_
    t_ax = get_parameter_ax(t_0, t_diff, t_num) if t_ax_ is None else t_ax_
    
    cost = np.zeros([len(a_ax), len(w_ax)])
    
    fig = None
    for a_idx, a_i in enumerate(a_ax):
        for w_idx, w_i in enumerate(w_ax):
            try:
                state_i = np.array([a_i, w_i, t_0])
                inversion.set_thermocline_spline(state_i)
                
                cost_i = inversion.Cost()
                cost[a_idx, w_idx] = cost_i
                
                print(f"{str((a_idx * len(w_ax) + w_idx + 1) / (len(a_ax) * len(w_ax))*100)[:4]}% | Cost = {cost_i:.4} | ({a_idx}, {w_idx})")
            except:
                errors.append([a_idx, a_i, w_idx, w_i])
                print(f"{str((a_idx * len(w_ax) + w_idx + 1) / (len(a_ax) * len(w_ax))*100)[:4]}% | FAILED | ({a_idx}, {w_idx})")

        fig = plot_cost_field(cost, a_ax, w_ax, a_0, w_0, fig=fig)
    
    return cost, errors, fig

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
test_oceanModel = OceanModel(_oceanmodel, sources, receivers, times, os.getcwd(), hv_points=(h0, v0), step_sizes=[0.0007, 0.0007, 0.0005], thermo_depth=250., thermo_amplitude=15., thermo_wave_length=500., verbose=False)

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
inversion = Inversion(test_oceanModel, verbose=False)
inversion.set_true_data(true_oceanModel)
inversion.know_the_real_answer(true_oceanModel.seabed_spline, true_oceanModel.get_thermocline_state())

cp_2 = log.checkpoint()


#cost, errors, fig = plot_cost_around_thermocline(inversion, true_oceanModel, a_diff=15, w_diff=499, a_num=2, w_num=2)#, a_ax_=np.linspace(0, 100, num=100))


#%%
print("Beginning inversion!")
"""
Solve seabed inversely
"""
best_c, best_model, best_idx, i = inversion.Solve(dv=1., dthermo=.02, alpha_seabed=4*10**6, alpha_thermo=10**4, max_iter=50, min_iter=0, thermocline_iter=60, plot_optimization=True)

cp_3 = log.checkpoint()
print("Ending inversion!")

#%%
def plot_switching_criteria(inversion):
    
    def get_S(cs):
        return np.array([np.std(cs[:i+1]) for i in range(len(cs))])
    
    def get_dS(cs):
        S = get_S(cs)
        return diff1d(S)
    
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)   
    
    cs = np.array(inversion.cs)
    S = get_S(cs)
    dS = get_dS(cs)
    iter_ax = np.arange(len(cs))
    ok = dS >= 0
    switch = np.argmin(ok[5:]) + 4
    
    ax0.plot(iter_ax, cs, '-r', linewidth=linewidth)
    ax0.plot(iter_ax[ok], cs[ok], '-g', linewidth=linewidth)
    ax0.set_ylabel("Error: E(i)",  fontsize=label_fontsize)
    ax0.axvline(2, linewidth=linewidth/2, color='black', ls=':', label='Initial skipped steps')
    ax0.axvline(switch, linewidth=linewidth/2, color='black', label='Point of switch')
    ax0.set_xlim([0, iter_ax.max()])
    ax1.plot(iter_ax, S, '-r', linewidth=linewidth)
    ax1.plot(iter_ax[ok], S[ok], '-g', linewidth=linewidth)
    ax1.set_ylabel("S(i)",  fontsize=label_fontsize)
#        ax1.set_ylabel("S(i) = std(E(:i))",  fontsize=label_fontsize)
    ax1.axvline(2, linewidth=linewidth/2, color='black', ls=':')
    ax1.axvline(switch, linewidth=linewidth/2, color='black')
    ax1.set_xlim([0, iter_ax.max()])
    ax2.plot(iter_ax, dS, '-r', linewidth=linewidth)
    ax2.plot(iter_ax[ok], dS[ok], '-g', linewidth=linewidth)
    ax2.set_ylabel(r"$\frac{dS}{di}$",  fontsize=title_fontsize)
    ax2.set_xlabel("Iterations (i)", fontsize=label_fontsize)
    ax2.axvline(2, linewidth=linewidth/2, color='black', ls=':')
    ax2.axvline(switch, linewidth=linewidth/2, color='black')
    ax2.axhline(0, linewidth=linewidth/2, color='black', ls='--')
    ax2.set_xlim([0, iter_ax.max()])
    ax0.set_title("Determining switching criteria during optimization", fontsize=title_fontsize)
    fig.legend(fontsize=label_fontsize)
    
    return fig
plot_switching_criteria(inversion)

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


