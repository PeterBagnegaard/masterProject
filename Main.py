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
    v = true_oceanModel.seabed_spline.coordinates()[1]
    data = np.zeros([len(v), len(dv)])
    
    for i, v_i in enumerate(v):
        for j, dv_j in enumerate(dv):
            new_v = np.copy(v)
            new_v[i] = v[i] + dv_j
            
            inversion.set_spline(new_v, target="seabed")
            data[i, j] = inversion.Cost()
            
            plt.clf()
            plt.imshow(data, aspect='auto', extent=[dv.min(), dv.max(), v.min(), v.max()])
            plt.colorbar()
            plt.title(f"i, j = {i}, {j}")
            plt.pause(0.01)
    return data

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
    figb = inversion.plot_inversion_history()
    manager = plt.get_current_fig_manager(); manager.full_screen_toggle()
    figc = inversion.plot_inversion_history_2()
    manager = plt.get_current_fig_manager(); manager.full_screen_toggle()
        
    plt.pause(2)
    figa.savefig('Switching_criteria.png', bbox_inches='tight',pad_inches = 0)
    figb.savefig('plot_inversion_history.png', bbox_inches='tight',pad_inches = 0)
    figc.savefig('plot_inversion_history_2.png', bbox_inches='tight',pad_inches = 0)

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

seabed_true = flat_noisy_coordinates(10, _oceanmodel.shape[1], 400, noise_fraction=0.1 )
seabed_test = flat_noisy_coordinates(10, _oceanmodel.shape[1], 400, noise_fraction=0)
#h, v, v_n = get_true_seabed(noise=0.1)

thermocline_true = flat_noisy_coordinates(10, _oceanmodel.shape[1], 200, noise_fraction=0)
thermocline_test = flat_noisy_coordinates(10, _oceanmodel.shape[1], 200, noise_fraction=0)

print("Creating ocean models")

#true_oceanModel = OceanModel(_oceanmodel, sources, receivers, times, os.getcwd(), hv_seabed_points=(h, v  ), hv_thermocline_points=thermocline_true, step_sizes=[0.0007, 0.0007, 0.0005], verbose=False)
#test_oceanModel = OceanModel(_oceanmodel, sources, receivers, times, os.getcwd(), hv_seabed_points=(h, v_n), hv_thermocline_points=thermocline_test, step_sizes=[0.0007, 0.0007, 0.0005], verbose=False, save_optimization=True)
true_oceanModel = OceanModel(_oceanmodel, sources, receivers, times, os.getcwd(), hv_seabed_points=seabed_true, hv_thermocline_points=thermocline_true, step_sizes=[0.0007, 0.0007, 0.0005], verbose=False)
test_oceanModel = OceanModel(_oceanmodel, sources, receivers, times, os.getcwd(), hv_seabed_points=seabed_test, hv_thermocline_points=thermocline_test, step_sizes=[0.0007, 0.0007, 0.0005], verbose=False, save_optimization=True)

true_oceanModel.initialize()

cp_1 = log.checkpoint()
print("Ocean models created")



#%%
"""
Initialize Inversion class
"""
inversion = Inversion(test_oceanModel, true_oceanModel, sigmas=get_sigma_matrix(true_oceanModel), verbose=True)

inversion.know_the_real_answer(true_oceanModel)


#%%
m_i = inversion.oceanModel.seabed_spline.coordinates()[1]

G = inversion.get_G(m_i, 3.)

C_M = np.diag(np.ones_like(m_i))
C_D = inversion.C_D
C_D_inv = inversion.C_D_inv

d_d = inversion.get_TOA() - inversion.true_toa
d_m = inversion.m_i() - inversion.m_priori
#%%
T = G.T @ C_D_inv
T_0 = T @ G
T_1 = T @ d_d
T_2 = np.linalg.inv(C_M) @ d_m
T_3 = np.linalg.inv(T_0 + C_M)
T_4 = T_1 + T_2
res = T_3 @ T_4


#%%
"""
Find G
"""
#def get_G():
def get_G(self, m_i, dv, cost0, target='seabed'):
    dv_ax = np.zeros_like(m_i)
    G = np.zeros([len(self.true_toa.copy()), len(m_i)])
    for i in range(len(m_i)):
        dv_ax[i-1] = 0.
        dv_ax[i] = dv
        m_new = m_i + dv_ax
        self.set_spline(m_new, target=target)
        d_i = self.oceanModel.TOA()
        G[:, i] = d_i - self.true_toa.copy()
    



#%%
"""
Solve inversion problem
"""
best_c, best_model, best_idx, i = inversion.Solve(variation_seabed=1, variation_thermocline=1, alpha_seabed=.03, alpha_thermo=0.05, seabed_iter=50, thermocline_iter=30, transition_iter=15, min_iter=3, plot_optimization=True, only_optimize_thermocline=False)


#%%

"""
Save oceanmodel figures 
"""
#save_oceanmodel_figures(true_oceanModel)

#%%
"""
Find best combination of variation_seabed and alpha_seabed

function (variation_seabed, alpha_seabed)
    return cs and inversion_history
    

[variation_seabed=0.16625, alpha=0.5635] => Cost = 14.158914
"""
def save_List(inversion, name):
    lines = [str(line) + '\n' for line in inversion.oceanModel.List]
    file = '/home/peter/Desktop/master_project/Madagascar/solve_res/List/' + name
    with open(file, 'a') as f:
        f.writelines(lines)
    inversion.oceanModel.List.clear()
    
def solve_res(variation_seabed, alpha_seabed, seabed_iter=50, transition_iter=15):
    # Create new true oceanModel
    true_oceanModel = OceanModel(_oceanmodel, sources, receivers, times, os.getcwd(), hv_seabed_points=flat_noisy_coordinates(10, _oceanmodel.shape[1], 400, noise_fraction=0.1 ), hv_thermocline_points=thermocline_true, step_sizes=[0.0007, 0.0007, 0.0005], verbose=False)
    # Initialize
    inversion = Inversion(test_oceanModel, true_oceanModel, sigmas=get_sigma_matrix(true_oceanModel), verbose=True)
    inversion.know_the_real_answer(true_oceanModel)
    t_0 = perf_counter()

    # Run simulation
    inversion.Solve(variation_seabed=variation_seabed, variation_thermocline=1, alpha_seabed=alpha_seabed, alpha_thermo=0.05, seabed_iter=seabed_iter, thermocline_iter=30, transition_iter=transition_iter, min_iter=3, plot_optimization=False, only_optimize_thermocline=False)
    
    # Save data
    time_list.append(perf_counter() - t_0)
    inversion_history_list.append(inversion.inversion_history)
    cs_list.append(inversion.cs)
    params.append([variation_seabed, alpha_seabed])
    # Restore to initial state
    inversion.inversion_history = []
    inversion.cs = []
    inversion.switch_idx = 0
    inversion.set_spline(seabed_initial)
    save_List(inversion, "v_" + str(variation_seabed) + "_a_" + str(alpha_seabed))


seabed_initial = np.copy(test_oceanModel.seabed_spline.coordinates()[1])
params = []
time_list = [] # 472 on average
cs_list = []
inversion_history_list = []
errors = []

variation_seabed_ax = 10 ** np.linspace(0, 0.7, num=7)
alpha_seabed_ax = 10 ** np.linspace(-1.4, -0.9, num=11)

for i, variation_seabed in enumerate(variation_seabed_ax):
    for j, alpha_seabed in enumerate(alpha_seabed_ax):
        try:
            print("==========================")
            print(f"{(len(variation_seabed_ax) * i + j) / (len(variation_seabed_ax) * len(alpha_seabed_ax)) * 100}%")
            print(f"{variation_seabed=}")
            print(f"{alpha_seabed=}")
            solve_res(variation_seabed, alpha_seabed)
            print(f"{len(cs_list[-1])} iterations took {time_list[-1]:.3}s")
            print("==========================")
        except Exception as ex:
            errors.append([[variation_seabed, alpha_seabed], ex])
        


save_res(params, time_list, cs_list, inversion_history_list, errors, folder_name='full_experiment_v_0.03_0.3_a_0.3_6.3')


#%%



#%%
def get_best_cs(cs_list):
    return np.array([min(cs) for cs in cs_list])
#    return np.array([cs[-1] for cs in cs_list])

def plot_best_cs(best_cs, ax=None, threshold=-1):
    A, V = np.meshgrid(alpha_seabed_ax, variation_seabed_ax)
    best_cs = best_cs.reshape([len(variation_seabed_ax), len(alpha_seabed_ax)])
#    best_cs = get_best_cs(cs_list).reshape([len(variation_seabed_ax), len(alpha_seabed_ax)])
    
    if threshold != -1:
        include = best_cs < threshold
        best_cs = best_cs * include + threshold * ~include
    
    if ax is None:
        _, ax = plt.subplots(1, 1)
    contour = ax.contourf(A, V, np.log10(best_cs))
    ax.figure.colorbar(contour)
    ax.set_ylabel("variation seabed")
    ax.set_xlabel("alpha seabed")
    return ax


def get_std(List, tail=5):
    return np.array([np.std(element[-tail:]) for element in List])

def get_std_of_seabed(inversion_history_list, s_true, mask=1, tail=3):
    def get_sqared_difference(e, s_true, mask):
        diff = e[mask:(10-mask)] - s_true[mask:(10-mask)]
        return np.sum(diff**2)
    
#    s_true = true_oceanModel.seabed_spline.coordinates()[1]
    
    diffs = np.zeros(len(inversion_history_list))
    for i, ih in enumerate(inversion_history_list):
        elements = ih[-tail:]
        diffs[i] = np.std([get_sqared_difference(e, s_true, mask) for e in elements])
    return diffs

def plot_seabed_mismatch(diffs, s_true, mask=1):
    
#    s_true = true_oceanModel.seabed_spline.coordinates()[1]
#    diffs = np.array([ih[-1][mask:(10-mask)] - s_true[mask:(10-mask)] for ih in inversion_history_list])
    
    misfits = np.sum(diffs**2, axis=1)
    mistfit_stds = get_std_of_seabed(inversion_history_list, s_true, mask=mask)
    
    
    fig, ax = plt.subplots(1, 1) 
    for i, (param, color) in enumerate(zip(variation_seabed_ax, colors[:n])):
        misfits_i = misfits[i*n:(i+1)*n]    
        sigma_i = mistfit_stds[i*n:(i+1)*n]
        ax.errorbar(alpha_seabed_ax, misfits_i, yerr=sigma_i, label=f"Variation = {param:.3}", linewidth=2)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel("Alpha")
    ax.set_ylabel("sum(true - estmiate)^2")
    ax.set_title("Error from true seabed vs alpha and variations")
    plt.legend()

def plot_misfit(cs, sigma_cs):
#    cs = get_best_cs(cs_list)
#    sigma_cs = get_std(cs_list)
    
    fig, ax = plt.subplots(1, 1) 
    for i, (param, color) in enumerate(zip(variation_seabed_ax, colors[:n])):
        cs_i = cs[i*n:(i+1)*n]
        sigma_i = sigma_cs[i*n:(i+1)*n]
        ax.plot(alpha_seabed_ax, cs_i, label=f"Variation = {param:.3}", linewidth=2)
    #    ax.errorbar(alpha_seabed_ax, cs_i, yerr=sigma_i, label=f"Variation = {param:.3}")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Misfit")
    ax.set_title("Misfit cs alpha and variations")
    plt.legend()

def plot_10_best_optimizations():
    fig, (ax_1d, ax_2d) = plt.subplots(1, 2)
    
    plot_best_cs(cs, ax_2d)
    
    #for i, (cs_i, color) in enumerate(zip(cs_sorted[:len(colors)], colors)):
    for i, color in zip(np.argsort(cs), colors):
        cs_i = cs_list[i]
        h, v = params[i]
        ax_1d.clear()
        ax_1d.semilogy(cs_i, color=color)
        ax_2d.plot(v, h, '*', color=color)
        plt.pause(1)
    

def plot_inversion_i(i, pause=0.1):
    v, a = params[best_idx]
    history_best = inversion_history_list[best_idx]
    cs_best = cs_list[best_idx]
    
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
    ax0.plot(h_true, s_true, '-b')
    fig.suptitle(f"Variation {v:.4}, Alpha {a:.4}")
    for point, color in zip(s_true, colors):
        print(point)
        ax2.axhline(point, color=color)
        
    for i, (history_i, cs_i) in enumerate(zip(history_best, cs_best)):
        
        ax0.plot(h_true, history_i[:len(h_true)], '.r', alpha = (i+1) / len(cs_best))
        ax1.semilogy(i, cs_i, '.k')
        plt.pause(pause)
        ax0.plot(h_true, history_i[:len(h_true)], '.k', alpha = ((i+1) / len(cs_best))**2)
        
        for point, color in zip(history_i, colors):
            ax2.plot(i, point, '.', color=color)
        
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
n = len(alpha_seabed_ax)

#%%

#%%


#%%
"""
Plot of misfit
"""
mask = 1
#best_cs = get_best_cs(cs_list).reshape([len(variation_seabed_ax), len(alpha_seabed_ax)])
cs = get_best_cs(cs_list)
sigma_cs = get_std(cs_list)
h_true, s_true = true_oceanModel.seabed_spline.coordinates()
diffs = np.array([ih[-1][mask:(10-mask)] - s_true[mask:(10-mask)] for ih in inversion_history_list])
best_idx = np.argmin(cs)

plt.close('all')

plot_seabed_mismatch(diffs, s_true)

plot_misfit(cs, sigma_cs)
    
plot_best_cs(cs)

plot_10_best_optimizations()

plot_inversion_i(best_idx)

#%%


#%%
"""
Solve inversion problem
"""
cp_2 = log.checkpoint()

print("Beginning inversion!")
"""
Solve seabed inversely
"""
best_c, best_model, best_idx, i = inversion.Solve(variation_seabed=1, variation_thermocline=1, alpha_seabed=.03, alpha_thermo=0.05, seabed_iter=50, thermocline_iter=30, transition_iter=15, min_iter=3, plot_optimization=True, only_optimize_thermocline=False)

cp_3 = log.checkpoint()
print("Ending inversion!")


#%%
#inversions_sorted = [inversion_history_list[i] for i in np.argsort(cs)[:10]]
#cs_sorted = [cs_list[i] for i in np.argsort(cs)[:10]]
#
#fig, (ax0, ax1) = plt.subplots(1, 2)
#for i, (inversion_i, cs_i) in enumerate(zip(inversions_sorted, cs_sorted)):
#    ax0.plot(cs_i, '.')
#    plt.pause(0.5)

#%%



#%%



#%%
"""
Save oceanmodel figures 
"""
save_inversion_figures(inversion)



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


