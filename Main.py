#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 14:15:01 2022

@author: peter
"""
from time import perf_counter
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from lib import (OceanModel,
                 Inversion,
                 log
                 )

def save_res(params, time_list, cs_list, inversion_history_list, errors, folder_name='experiment_v_0.01_2.5_a_0.001_9'):
    path = "/home/peter/Desktop/master_project/Madagascar/solve_res/"
    content = os.listdir(path)
    if folder_name in content:
        raise Exception (f"Folder {folder_name} already exists!")
    
    os.mkdir(path + folder_name)

    with open(path + folder_name + "/params", "wb") as fp:   #Pickling
        pickle.dump(params, fp)
    
    with open(path + folder_name + "/time_list", "wb") as fp:   #Pickling
        pickle.dump(time_list, fp)
    
    with open(path + folder_name + "/cs_list", "wb") as fp:   #Pickling
        pickle.dump(cs_list, fp)
    
    with open(path + folder_name + "/inversion_history_list", "wb") as fp:   #Pickling
        pickle.dump(inversion_history_list, fp)
    
    with open(path + folder_name + "/errors", "wb") as fp:   #Pickling
        pickle.dump(errors, fp)


def cost_around_true_thermocline(diff=10, num=50):
    dv = np.linspace(-diff, diff, num=num)
    h, v = true_oceanModel.thermocline_spline.coordinates()
    data = np.zeros([len(v), len(dv)])
    
    for i, v_i in enumerate(v):
        for j, dv_j in enumerate(dv):
            new_v = np.copy(v)
            new_v[i] = v_i + dv_j
            
            inversion.set_spline(new_v, target="thermocline")
            data[i, j] = inversion.Cost()
            print(new_v)
            print(f"{dv_j} -> {data[i, j]}")
            
            plt.clf()
            plt.imshow(np.log10(data.T), aspect='auto', extent=[h.min(), h.max(), dv.min(), dv.max()])
            plt.colorbar()
            plt.ylabel("Variation from true value")
            plt.xlabel("Horizontal position")
            plt.title(f"{i+1} of {len(v)}, {j+1} of {len(dv)}")
            plt.pause(0.01)
    inversion.set_spline(v, target="thermocline")
    return data

def compare_ocean_models(true_oceanModel, test_oceanModel):
    plt.figure()
    true_thermo = true_oceanModel.thermo_state()
    test_thermo = test_oceanModel.thermo_state()
    plt.subplot(1, 3, 1)
    plt.hist(true_thermo - test_thermo)    
    plt.title("Difference between thermocline parameters")
    print("\nDifference between thermocline parameters")
    print(true_thermo - test_thermo)

    true_seabed = true_oceanModel.seabed_state()
    test_seabed = test_oceanModel.seabed_state()
    plt.subplot(1, 3, 2)
    plt.hist(true_seabed - test_seabed)
    plt.title("Difference between seabed points")
    print("\nDifference between seabed points")
    print(true_seabed - test_seabed)
    
    toa_true = true_oceanModel.TOA()
    toa_test = test_oceanModel.TOA()
    
    print(f"\n{(toa_test == toa_true).sum()} of {len(toa_true)} TOA are identical")
        
    plt.subplot(1, 3, 3)
    plt.hist(toa_true / toa_test, bins=20)
    plt.title("TOA ratio")

def save_inversion_figures(inversion, v=17., P=None):
    if P is None:
        P = inversion.posteriori_covariance(v)[0]
    figa = inversion.plot_end_result(true_oceanModel, P, v)
    manager = plt.get_current_fig_manager(); manager.full_screen_toggle()
    figb = inversion.plot_inversion_history()
    manager = plt.get_current_fig_manager(); manager.full_screen_toggle()
    figc = inversion.plot_posteriori_covariance(P, v)
    manager = plt.get_current_fig_manager(); manager.full_screen_toggle()
    figd = inversion.plot_reflection_points(P, v)
    manager = plt.get_current_fig_manager(); manager.full_screen_toggle()
    
#    figc = inversion.plot_inversion_history_2()
#    manager = plt.get_current_fig_manager(); manager.full_screen_toggle()
        
    plt.pause(5)
    figa.savefig('plot_end_result.png', bbox_inches='tight',pad_inches = 0)
    figb.savefig('plot_inversion_history.png', bbox_inches='tight',pad_inches = 0)
    figc.savefig('plot_posteriori_covariance.png', bbox_inches='tight',pad_inches = 0)
    figd.savefig('plot_reflection_points.png', bbox_inches='tight',pad_inches = 0)

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

#def get_true_data():
#    """
#    Load OceanModel.mat to np array
#    """
#    _true_oceanmodel = loadmat('OceanModel.mat')['OceanModel'].astype(np.float32)
#    _true_oceanmodel = gaussian_filter(_true_oceanmodel, 5)
#    return _true_oceanmodel

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

def flat_noisy_coordinates(n_spline, h_span, seabed_height, noise_fraction=0.):
    h_ax = np.linspace(0, h_span, num=n_spline) #
    v_ax = np.ones_like(h_ax) * seabed_height
    noise = v_ax * np.random.randn(n_spline) * float(noise_fraction)
    return (h_ax, v_ax + noise)

def set_true_seabed(inversion, true_oceanModel, scrample=False):
    points = true_oceanModel.seabed_spline.coordinates()[1]
    if scrample:
        points[5] = 1.1*points[5]
    inversion.set_spline(points, target='seabed')

def get_true_seabed(noise=0.01):
    h = np.array([   0.        ,  220.55555556,  441.11111111,  661.66666667,
             882.22222222, 1102.77777778, 1323.33333333, 1543.88888889,
            1764.44444444, 1985.        ])
    v = np.array([376.07856812, 437.47310362, 405.49548887, 399.50116633,
            393.17192563, 357.28176729, 421.62255262, 378.05737762,
            383.66654188, 392.45178946])
    v_noise = np.random.randn(len(v)) * float(noise) * v.mean()
    return h, v, v + v_noise

def get_model_error(scaling=1.):
    return np.ones(len(true_oceanModel.seabed_state())) * scaling
#    return scaling * np.array([305.95466769,  19.90975113,   9.61094035,   6.13700691, 5.57069161,   5.57069161,   6.13700691,   9.61094035, 19.90975113, 305.95466769])
#    return scaling * np.array([305.95466769,  19.90975113,   9.61094035,   6.13700691, 5.57069161,   5.57069161,   6.13700691,   9.61094035, 19.90975113, 305.95466769])

def get_model_error_thermo(scaling=1.):
    return np.ones(len(true_oceanModel.thermo_state())) * scaling
#    return scaling * np.array([305.95466769,  14., 14., 305.95466769])

def get_pos(n=10, width=1985, depth=142, offset=0):
    h = np.linspace(0, width, n+2) + offset
    h = h[1:-1]
    v = np.ones_like(h) * depth
    return np.array([[h_i, v_i] for h_i,v_i in zip(h, v)])

plt.close('all')
os.chdir('/home/peter/Desktop/master_project/Madagascar/MadClass')

np.random.seed(42)

# Set what to log
log.log_function_calls_mode = False
log.log_function_counts_mode = False

"""
Create OceanModel Classes
coordinates [horizontal, vertical]
"""

_oceanmodel = get_uniform_field([500, 1985])

seabed_true = flat_noisy_coordinates(10, _oceanmodel.shape[1], 400, noise_fraction=0.1)
seabed_test = flat_noisy_coordinates(10, _oceanmodel.shape[1], 400, noise_fraction=0)
#h, v, v_n = get_true_seabed(noise=0.1)
thermocline_true = flat_noisy_coordinates(4, _oceanmodel.shape[1], 200, noise_fraction=0.1)
thermocline_test = flat_noisy_coordinates(4, _oceanmodel.shape[1], 200, noise_fraction=0)

#seabed_test = seabed_true
#thermocline_test = thermocline_true

#sources   = [[i, 0.1] for i in np.linspace(0.1, 1.3, num=10)]
#receivers = [[i, 0.1] for i in np.linspace(0.1, 1.3, num=10)+0.05]
#sources   = [[i, 0.1 / 0.0007] for i in (np.linspace(0.1, 1.3, num=10)) / 0.0007]
#receivers = [[i, 0.1 / 0.0007] for i in (np.linspace(0.1, 1.3, num=10)+0.05) / 0.0007]
sources = get_pos(offset=40)
receivers = get_pos(offset=-40)

#thermocline_test[1][5] = 215 #!!!
print("Creating ocean models")

scaling = 10.
true_oceanModel = OceanModel(_oceanmodel, sources*scaling, receivers*scaling, os.getcwd(), hv_seabed_points=seabed_true, hv_thermocline_points=thermocline_true, step_sizes=[scaling, scaling, scaling], verbose=True)
test_oceanModel = OceanModel(_oceanmodel, sources*scaling, receivers*scaling, os.getcwd(), hv_seabed_points=seabed_test, hv_thermocline_points=thermocline_test, step_sizes=[scaling, scaling, scaling], verbose=False, save_optimization=True)

"""
Get true toa and clean up afterwards
"""
true_oceanModel.initialize()
true_toa = true_oceanModel.TOA()
true_oceanModel.plot_oceanmodel()
clean_rsf()
#test_oceanModel.initialize()
#test_oceanModel.plot_source_receiver(0, 9)
#toa1 = test_oceanModel.TOA()

cp_1 = log.checkpoint()
print("Ocean models created")

#%%
"""
Initialize Inversion class
"""
inversion = Inversion(test_oceanModel, true_toa, sigmas=get_sigma_matrix(true_oceanModel, sigma=0.01), C_M=get_model_error(500), C_M_thermo=get_model_error_thermo(500), verbose=True)

inversion.know_the_real_answer(true_oceanModel)


#%%
"""
Solve inversion problem
"""
a = 1.
v = 17.
best_c, best_model, best_idx, i = inversion.Solve(variation_seabed=v, 
                                                  variation_thermocline=v, 
                                                  alpha_seabed=a, 
                                                  alpha_thermo=a, 
                                                  seabed_iter=5, 
                                                  thermocline_iter=10, 
                                                  transition_iter=0, 
                                                  min_iter=0, 
                                                  plot_optimization=True, 
                                                  only_optimize_thermocline=False)

#inversion.plot_inversion_history_2()

#%%
plt.close('all')
import matplotlib.gridspec as gridspec
cmap=plt.get_cmap('Blues')
label_fontsize = 17
title_fontsize = 20
tick_fontsize = 20
markersize_big = 20
markersize = 15
markersize_small = 6
linewidth = 5
linewidth_thin = 2.5
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
def set_tick_fontsize(ax):
    [tick.set_fontsize(tick_fontsize) for tick in ax.get_xticklabels()]
    [tick.set_fontsize(tick_fontsize) for tick in ax.get_yticklabels()]


if len(inversion.cs) == 0:
    raise Exception("No inversion history found!")
    
n = len(inversion.seabed_horizontal)
switch = inversion.switch_idx + inversion.transition_iter

data = np.array(inversion.inversion_history)
seabeds = data[:, :n]
thermos = data[:, n:]
cs = np.array(inversion.cs)
seabeds_ax = np.arange(0, seabeds.shape[0])

gs = gridspec.GridSpec(nrows=8, ncols=1)
fig = plt.figure()
ax0 = fig.add_subplot(gs[0:3, 0])
ax1 = fig.add_subplot(gs[3:6, 0], sharex=ax0)
ax2 = fig.add_subplot(gs[6:8, 0], sharex=ax1)
ax0.xaxis.set_visible(False)
ax1.xaxis.set_visible(False)

ax0.axvline(switch, color='black')
for i, c in enumerate(colors[:thermos.shape[1]]):
    ax0.plot(seabeds_ax, thermos[:, i], '--D', color=c, linewidth=linewidth, markersize=markersize)
for j, (t, c) in enumerate(zip(inversion.true_thermocline, colors[:seabeds.shape[1]])):
    ax0.plot([seabeds_ax[-1], seabeds_ax[-1]*1.05], [thermos[-1, j], t], '-D', color=c, linewidth=linewidth, markersize=markersize, label=f"point {j}")
ax0.text(seabeds_ax[-1]*1.07, inversion.true_thermocline.mean()*1.08, "True thermocline", rotation=-90, fontsize=title_fontsize)

ax1.axvline(switch, color='black')
for i, c in enumerate(colors[:seabeds.shape[1]]):
    ax1.plot(seabeds_ax, seabeds[:, i], '--s', color=c, linewidth=linewidth, markersize=markersize)
for j, (s, c) in enumerate(zip(inversion.true_seabed, colors[:seabeds.shape[1]])):
    ax1.plot([seabeds_ax[-1], seabeds_ax[-1]*1.05], [seabeds[-1, j], s], '-s', color=c, linewidth=linewidth, markersize=markersize, label=f"point {j}")
ax1.text(seabeds_ax[-1]*1.07, inversion.true_seabed.mean()*1.08, "True ocean floor", rotation=-90, fontsize=title_fontsize)

ax2.axvline(switch, color='black', label="Switch")
ax2.semilogy(seabeds_ax, cs, '-o', linewidth=linewidth, markersize=markersize, color='black')
ax2.set_ylabel("Misfit", fontsize=label_fontsize)
ax2.set_xlabel("Iteration", fontsize=label_fontsize)

#fig.tight_layout()
fig.subplots_adjust(hspace=0)
ax0.invert_yaxis()
ax1.invert_yaxis()
fig.legend(fontsize=label_fontsize, loc='center right')
ax0.set_ylabel(r"hermocline optimization" + "\n" + r"Depth ($v$)", fontsize=label_fontsize)
ax1.set_ylabel(r"Seabed optimization" + "\n" + "Depth ($v$)", fontsize=label_fontsize)
ax0.set_title("Invertion process", fontsize=title_fontsize)
set_tick_fontsize(ax0)
set_tick_fontsize(ax1)
set_tick_fontsize(ax2)




#%%

"""
Save oceanmodel figures 
"""
save_oceanmodel_figures(true_oceanModel)


#%%
"""
Save oceanmodel figures 
"""
save_inversion_figures(inversion, v, P)




#%%
"""

"""
def reset():
    inversion.set_spline(seabed_test[1], "seabed")
    inversion.set_spline(thermocline_test[1], "thermocline")
    inversion.cs = []
    inversion.inversion_history = []
    inversion.errors = []
    plt.close('all')

def save_inversion(inversion, params, path):
    a, v = params
    cs_list = inversion.cs
    inversion_history_list = inversion.inversion_history
    errors = inversion.errors
    folder_name = f"/{a=}_{v=}"
    
#    path = "/home/peter/Desktop/master_project/Madagascar/solve_inversion/"
    content = os.listdir(path)
    if folder_name in content:
        raise Exception (f"Folder {folder_name} already exists!")
    
    os.mkdir(path + folder_name)

    with open(path + folder_name + "/params", "wb") as fp:   #Pickling
        pickle.dump(params, fp)
    
    with open(path + folder_name + "/cs_list", "wb") as fp:   #Pickling
        pickle.dump(cs_list, fp)
    
    with open(path + folder_name + "/inversion_history_list", "wb") as fp:   #Pickling
        pickle.dump(inversion_history_list, fp)
    
    with open(path + folder_name + "/errors", "wb") as fp:   #Pickling
        pickle.dump(errors, fp)


#v_ax = np.linspace(1, 25, num=10)
#a_ax = np.linspace(1, 3, num=7)
v_ax = np.linspace(11.5, 22.5, num=12)
#a_ax = np.linspace(0.4, 1.6, num=4)
a_ax = [1.]

woops_errors = []
path = "/home/peter/Desktop/master_project/Madagascar/variation_newton_method"
content = os.listdir(path)

for a in a_ax:
    for v in v_ax:
        print("================================")
        print(f"{a=}, {v=}")
        print("================================")
        try:
            content = os.listdir(path)
            folder_name = f"{a=}_{v=}"
            if folder_name in content:
                print(f"{folder_name} already exists!")
                woops_errors.append(f"Skipping {folder_name}")
                continue
            
            reset()
            
            inversion.Solve(  variation_seabed=v, 
                              variation_thermocline=v, 
                              alpha_seabed=a, 
                              alpha_thermo=a, 
                              seabed_iter=10, 
                              thermocline_iter=25, 
                              transition_iter=1, 
                              min_iter=0, 
                              plot_optimization=True, 
                              only_optimize_thermocline=False)
            
            save_inversion(inversion, (a, v), path)
        except Exception as e:
            print(e)
            woops_errors.append([a, v, e])







#%%
"""
Print names of unised functions
"""
unused_functions = []
used_functions = list(cp_1.function_counts.keys()) #+ list(cp_2.function_counts.keys()) + list(cp_3.function_counts.keys())
for func_name in log.functions:
    if not func_name in used_functions:
        unused_functions.append(func_name)
print(unused_functions)

#%%
import matplotlib.gridspec as gridspec
cmap=plt.get_cmap('Blues')
label_fontsize = 15
title_fontsize = 20
tick_fontsize = 20
markersize_big = 20
markersize = 15
markersize_small = 6
linewidth = 5
linewidth_thin = 2.5
tick_fontsize = 15
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']




original_s = inversion.state('seabed')
h_10, s_10 = inversion.oceanModel.seabed_spline.coordinates()
smush = (h_10[1] - h_10[0]) * 0.06
h_cheat = h_10.copy()
h_cheat[0] += smush
h_cheat[-1] -= smush

y, x = np.histogram(inversion.oceanModel.List, bins=50)

h_ax = inversion.oceanModel.dense_horizontal_ax
v_ax_test = inversion.oceanModel.seabed_spline(h_ax, True)
inversion.set_spline(inversion.true_seabed)
v_ax_true = inversion.oceanModel.seabed_spline(h_ax, True)
inversion.set_spline(original_s)

sigma_seabed = abs(P[:len(original_s), :]).sum(axis=1) ** 0.5

fig = plt.figure()
fig.tight_layout()
fig.subplots_adjust(hspace=0)
gs = gridspec.GridSpec(4, 1)

ax0 = fig.add_subplot(gs[0:2, 0])
ax0.bar((x[1:] + x[:-1])/2, y, width = (x[1]-x[0])*0.9, color='black')
ax0.xaxis.set_visible(False)
ax0.set_ylabel("Count", fontsize=label_fontsize)
ax0.set_title("Points of reflection on seabed", fontsize=title_fontsize)
#set_tick_fontsize(ax0)
ax0.set_xlim([h_ax[0], h_ax[-1]])

ax1 = fig.add_subplot(gs[2, 0])
#ax1.plot(h_ax, v_ax_initial, '-k', linewidth=linewidth, label="Initial seabed")
ax1.plot(h_ax, v_ax_true, '-k', linewidth=linewidth, label="True seabed")
ax1.plot(h_ax, v_ax_test, '-r', linewidth=linewidth, label="Recovered seabed")
ax1.invert_yaxis()
ax1.xaxis.set_visible(False)
ax1.set_ylabel(r"Depth ($v$)", fontsize=label_fontsize)
ax1.legend(fontsize=label_fontsize)
#set_tick_fontsize(ax1)
ax1.set_xlim([h_ax[0], h_ax[-1]])

ax2 = fig.add_subplot(gs[3, 0])
ax2.plot(h_ax, np.zeros_like(h_ax), '-k', linewidth=linewidth_thin)
ax2.plot(h_ax, v_ax_true - v_ax_test, '-r', linewidth=linewidth, label="Error")
ax2.errorbar(h_cheat, inversion.true_seabed - s_10, yerr=sigma_seabed, color='red', linewidth=linewidth, label="Error", ls='None', zorder=10)
ax2.invert_yaxis()
ax2.set_xlabel(r"Horizontal distance ($h$)", fontsize=label_fontsize)
ax2.set_ylabel(r"Depth ($v$)", fontsize=label_fontsize)
ax2.legend(fontsize=label_fontsize)
#set_tick_fontsize(ax2)
ax2.set_xlim([h_ax[0], h_ax[-1]])
