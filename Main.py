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

def get_model_error():
    return 10**2 / np.array([305.95466769,  19.90975113,   9.61094035,   6.13700691, 5.57069161,   5.57069161,   6.13700691,   9.61094035, 19.90975113, 305.95466769])

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

seabed_true = flat_noisy_coordinates(10, _oceanmodel.shape[1], 400, noise_fraction=0.01 )
seabed_test = flat_noisy_coordinates(10, _oceanmodel.shape[1], 400, noise_fraction=0)
#h, v, v_n = get_true_seabed(noise=0.1)

thermocline_true = flat_noisy_coordinates(10, _oceanmodel.shape[1], 200, noise_fraction=0)
thermocline_test = flat_noisy_coordinates(10, _oceanmodel.shape[1], 200, noise_fraction=0)
thermocline_test[1][5] = 215 #!!!
print("Creating ocean models")

#true_oceanModel = OceanModel(_oceanmodel, sources, receivers, times, os.getcwd(), hv_seabed_points=(h, v  ), hv_thermocline_points=thermocline_true, step_sizes=[0.0007, 0.0007, 0.0005], verbose=False)
#test_oceanModel = OceanModel(_oceanmodel, sources, receivers, times, os.getcwd(), hv_seabed_points=(h, v_n), hv_thermocline_points=thermocline_test, step_sizes=[0.0007, 0.0007, 0.0005], verbose=False, save_optimization=True)
true_oceanModel = OceanModel(_oceanmodel, sources, receivers, times, os.getcwd(), hv_seabed_points=seabed_true, hv_thermocline_points=thermocline_true, step_sizes=[0.0007, 0.0007, 0.0005], verbose=False)
test_oceanModel = OceanModel(_oceanmodel, sources, receivers, times, os.getcwd(), hv_seabed_points=seabed_test, hv_thermocline_points=thermocline_test, step_sizes=[0.0007, 0.0007, 0.0005], verbose=False, save_optimization=True)

true_oceanModel.initialize()

cp_1 = log.checkpoint()
print("Ocean models created")



"""
Initialize Inversion class
"""
inversion = Inversion(test_oceanModel, true_oceanModel, sigmas=get_sigma_matrix(true_oceanModel), C_M=get_model_error(), verbose=True)

inversion.know_the_real_answer(true_oceanModel)

#%%
j = inversion.derivative(inversion.state("thermocline"), 4, 0, target='thermocline')

_, ax = plt.subplots(2, 1)
ax[0].plot(j , '.')
ax[1].plot(inversion.state("thermocline") , '.')

#%%
"""
Solve inversion problem
"""
#a=4.117647058823529
#v=-3.8235294117647056
#a= 0.5
#v = 3.

#inversion.a = a
#inversion.v = v
best_c, best_model, best_idx, i = inversion.Solve(variation_seabed=4, 
                                                  variation_thermocline=1, 
                                                  alpha_seabed=1., 
                                                  alpha_thermo=0.05, 
                                                  seabed_iter=50, 
                                                  thermocline_iter=30, 
                                                  transition_iter=15, 
                                                  min_iter=3, 
                                                  plot_optimization=True, 
                                                  only_optimize_thermocline=False)

#cov = inversion.posteriori_covariance(v)

#%%



#%%

fig, ax = plt.subplots(1, 1)
for covariance, t_data, t_model in inversion.data:
    ax.clear()
    
    err = covariance.diagonal()
    ax.errorbar(inversion.seabed_horizontal-10, covariance @ t_data , yerr=err, label="data", color='red')
    ax.errorbar(inversion.seabed_horizontal+10, covariance @ t_model, yerr=err, label="model", color='blue')
    ax.legend()
    
    plt.pause(0.4)
    #%%

fig, ax = plt.subplots(1, 1)
n = len(inversion.data)
dh = inversion.seabed_horizontal[1] - inversion.seabed_horizontal[0]
for i, (covariance, t_data, t_model) in enumerate(inversion.data):   
    err = covariance.diagonal()
    ax.errorbar(inversion.seabed_horizontal + i*dh/n, covariance @ t_data , yerr=err, label="data" , color='red' , alpha=i/n)
#    ax.errorbar(inversion.seabed_horizontal + i*dh/n, covariance @ t_model, yerr=err, label="model", color='blue', alpha=i/n)
ax.legend()
    

#%%
a=4.117647058823529
v=-3.8235294117647056

inversion.set_spline(seabed_test[1])
m_i = inversion.m_i()

grad_old = inversion.derivative(m_i, v, inversion.Cost()) * 0.03
inversion.set_spline(seabed_test[1])
grad_new = inversion.quasi_Newton(v) * a
#%%
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(true_oceanModel.m_i(), ':r', label="True")
plt.plot(inversion.oceanModel.m_i(), ':b', label="Test")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(grad_old, ':r', label="old")
plt.plot(-grad_new, ':b', label="new")
plt.legend()
#%%

d_data = inversion.get_TOA() - true_oceanModel.TOA()
d_model = inversion.m_i() - true_oceanModel.m_i()
d_data = d_data.reshape([-1, 1])
d_model = d_model.reshape([-1, 1])

plt.subplot(2, 2, 1)
plt.hist(d_data)
plt.title("Delta data")
plt.subplot(2, 2, 2)
plt.hist(d_model)
plt.title("Delta model")
plt.subplot(2, 2, 3)
plt.hist(inversion.C_D_inv.diagonal())
plt.subplot(2, 2, 4)
plt.hist(inversion.C_M_inv.diagonal())

error_data = d_data.T @ inversion.C_D_inv @ d_data
error_model = d_model.T @ inversion.C_M_inv @ d_model

print(f"{error_data=}")
print(f"{error_model=}")

#%%
def mean_std(v):
    diff = true_oceanModel.m_i() - inversion.m_i()
    grad = inversion.quasi_Newton(v)
    mean = np.mean(grad[1:-1] / diff[1:-1])
    std = np.std(grad[1:-1] / diff[1:-1])
    
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(diff, ':r', label="True")
    plt.plot(grad, ':b', label="Gradient")
    plt.legend()
    plt.title(v)
    plt.subplot(2, 1, 2)
    plt.plot(grad / diff)
    plt.axhline(mean, color='black')
    plt.axhline(mean+std, color='gray')
    plt.axhline(mean-std, color='gray')
    return mean, std

means = []
stds = []
v_ax = np.linspace(-4, 4, num=20)
for v in v_ax:
    print(v)
    mean, std = mean_std(v)
    means.append(mean)
    stds.append(std)
    plt.pause(0.01)
    
plt.figure()
plt.errorbar(v_ax, means, yerr=stds)
plt.axhline(0, color='black')
plt.xlabel("Variation")
plt.ylabel("gradient / true gradient")
#%%
"""
Calculate step
"""
#v=1
#
#m_i = inversion.m_i()
#
#G = inversion.get_G(m_i, v)    # +- 10^-5
#
#C_M_inv = np.linalg.inv(np.diag(get_model_error()))
#C_D_inv = inversion.C_D_inv
#
#d_d = inversion.get_TOA() - inversion.true_toa
#d_m = inversion.m_i() - inversion.m_priori
#
#T = G.T @ C_D_inv                   # +- 10^7
#T_0 = T @ G                         # +  7000
#T_1 = T @ d_d                       # +  7000
#T_2 = C_M_inv @ d_m                 #    0
#T_3 = np.linalg.inv(T_0 + C_M_inv)  # +- 0.4
#T_4 = T_1 + T_2                     # +  7000
#res = T_3 @ T_4                     # +  0.5


#%%

plt.close('all')
def get_a_cost(grad):
    cost = []
    ax0.plot(m_i, '-k')
    ax0.plot(true_oceanModel.m_i(), '-r')
    print(m_i)
    for a in a_ax:
        inversion.set_spline(m_i - a*grad)
        cost.append(inversion.Cost() / cost0)
        print(f"a = {a} | {cost[-1]}")
        
        ax1.clear()
        ax0.plot(inversion.m_i(), ':k')
        ax1.plot(a_ax[:len(cost)], cost)
        plt.pause(0.01)
    inversion.set_spline(m_i)
    return cost


a_ax = np.linspace(-10, 10, num=18)
v_ax = np.linspace(-5, 5, num=18)
A, V = np.meshgrid(a_ax, v_ax)
J = np.zeros_like(A)
costs = []
cost_best = []

inversion.set_spline(seabed_test[1])
m_i = inversion.m_i().copy()
cost0 = inversion.Cost()

_, (ax0, ax1) = plt.subplots(1, 2)
_, ax2 = plt.subplots(1, 1)
for i, v in enumerate(v_ax):
    ax0.clear()
    ax1.clear()
    print(f"======={v}=======")
    grad = inversion.quasi_Newton(v)
    cost = get_a_cost(grad)
    costs.append(cost)
    J[i, :] = np.array(cost)
    ax2.clear()
    ax2.contourf(A, V, J)
    ax2.set_xlabel("a")
    ax2.set_ylabel("v")
    plt.pause(0.1)

#%%
_, ax3 = plt.subplots(1, 1)  
JJ = J.copy()
JJ[JJ > 1] = 1
im = ax3.contourf(A, V, JJ)
ax3.set_xlabel("a")
ax3.set_ylabel("v")
plt.colorbar(im)

for _ in range(10):
    i, j = np.unravel_index(JJ.argmin(), JJ.shape)
    JJ[i, j] = 1.
    plt.plot(a_ax[j], v_ax[i], '*')
    plt.pause(0.5)
    print(f"a={a_ax[j]} | v={v_ax[i]}")

#%%
import numpy as np

def make_field(price, vertical_pixels=10):
    y_axis = np.linspace(price.min(), price.max(), num=vertical_pixels)
    data = np.zeros([vertical_pixels, len(price)])
    for i, p in enumerate(price):
        idx = np.argmin(abs(y_axis - p))
        data[idx, i] = 1
    return data

#%%
A, V = np.meshgrid(a_ax, v_ax)

j = np.array(costs)/935.162520790935

plt.close('all')
plt.figure()
plt.contourf(A, V, np.log10(j))
#plt.contourf(A[:3, :], V[:3, :], np.log10(j[:3, :]))
plt.colorbar()

plt.figure()
#for v_i, cost_v in zip(v_ax[:3], costs[:3]):
plt.axvline(0, color='black')
for v_i, cost_v in zip(v_ax, costs):
    plt.semilogy(a_ax, np.array(cost_v)/935.162520790935, '.:', label=f"v={v_i}")
    plt.title(v_i)
    plt.xlabel("a")
#    plt.pause(1.3)
plt.legend()
#%%



#%%
inversion.set_spline(seabed_test[1])

cs = [inversion.Cost()] # 24360.913009563545
history = [inversion.m_i()]
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

while True:
    step = inversion.quasi_Newton(v=v, a=a)
    print(abs(step).mean())
    m_new = inversion.m_i() - step
    inversion.set_spline(m_new)
    cs.append(inversion.Cost())
    history.append(inversion.m_i())

    plt.clf()
    plt.subplot(1, 2, 1)
    plt.plot(cs, '.')
    plt.subplot(1, 2, 2)
    j = np.array(history)
    for i, color in enumerate(colors):
        plt.plot(j[:, i], ':', color=color, label=i)
        plt.axhline(true_oceanModel.m_i()[i], color=color)
    plt.legend()
    plt.pause(0.01)

#%%
cost0 = inversion.Cost()
derivative = inversion.derivative(m_i, 3, cost0)




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


