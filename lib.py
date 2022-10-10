from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import PchipInterpolator
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec
#from scipy.optimize import minimize
from scipy.signal import convolve
import matplotlib.pyplot as plt
from time import perf_counter
from scipy.io import loadmat
from numba import jit
import rsf.api as rsf
import numpy as np
import functools
import m8r
import os
import re

cmap=plt.get_cmap('Blues')
label_fontsize = 15
title_fontsize = 20
tick_fontsize = 20
markersize = 15
linewidth = 5
linewidth_thin = 2.5
tick_fontsize = 15
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

#%%
"""
Decorators
"""
def timer_decorator(func, print_mode=True):
    
    @functools.wraps(func)
    def timer_decorator_wrapper(*args, **kwargs):
        start_time = perf_counter()
        res = func(*args, **kwargs)
        lapsed_time = perf_counter() - start_time
        if print_mode:
            print(f"{func.__name__} took {lapsed_time} to run")
            return res
        
        return res, lapsed_time
    
    return timer_decorator_wrapper

class Log:
    
    def __init__(self):
        self.function_calls = []
        self.function_counts = {}
        self.function_times = {}
        self.functions = []
        self.log_function_calls_mode = False
        self.log_function_counts_mode = True
        self.log_function_time_mode = True
    
    def __call__(self, func):
        self.functions.append(self.get_name(func))
                
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.log_function_calls(func, *args, **kwargs)
            self.log_function_counts(func)
            res = self.log_function_time(func, *args, **kwargs)
            if not self.log_function_time_mode:
                return func(*args, **kwargs)
            else:
                return res
        
        return wrapper
    
    def log_function_time(self, func, *args, **kwargs):
        if self.log_function_time_mode:
            start_time = perf_counter()
            res = func(*args, **kwargs)
            lapsed_time = perf_counter() - start_time
            
            func_name = self.get_name(func)
            if func_name in self.function_times:
                self.function_times[func_name] += lapsed_time
            else:
                self.function_times[func_name] = lapsed_time
            return res

    
    def log_function_counts(self, func):
        if (self.log_function_counts_mode):
            func_name = self.get_name(func)
            if func_name in self.function_counts:
                self.function_counts[func_name] += 1
            else:
                self.function_counts[func_name] = 1
            
    def log_function_calls(self, func, *args, **kwargs):
        if (self.log_function_calls_mode):
            args_repr = [repr(a) for a in args]
            kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
            signature = ", ".join(args_repr + kwargs_repr)
            func_logstring = f"Calling {self.get_name(func)}({signature})"
            self.function_calls.append(func_logstring)
            
    def get_name(self, func):
        return func.__qualname__
    
    def checkpoint(self):
        checkpoint = Log()
        checkpoint.function_calls = self.function_calls
        checkpoint.function_counts = self.function_counts
        checkpoint.function_times = self.function_times
        self.reset()
        return checkpoint
    
    def reset(self):
        self.function_calls = []
        self.function_counts = {}
        self.function_times = {}
        
    """ PLOTTING """
    def plot_calls(self, yscale='log'):
        if len(self.function_counts) == 0:
            raise Exception("No function calls have been logged!")
        
        keys, vals = self.function_counts.keys(), self.function_counts.values()
        keys = list(keys)
        vals = np.array(list(vals))
               
        sort_idx = np.argsort(vals)[::-1]
        keys_sorted = [keys[i] for i in sort_idx]
        vals_sorted = [vals[i] for i in sort_idx]
        
        x_ax = np.arange(len(vals))

        fig, ax = plt.subplots(1, 1)
        ax.bar(x_ax, vals_sorted, color='black')
        ax.set_yscale(yscale)
        
        ax.set_ylabel("# Function calls", fontsize=label_fontsize)
        ax.set_xticks(x_ax)
        ax.set_xticklabels(keys_sorted, rotation = 90)
        
        plt.subplots_adjust(bottom=0.42)

    def plot_time(self, yscale='log'):
        if len(self.function_times) == 0:
            raise Exception("No function times have been logged!")
        
        keys, vals = self.function_times.keys(), self.function_times.values()
        keys = list(keys)
        vals = np.array(list(vals))
        
        sort_idx = np.argsort(vals)[::-1]
        keys_sorted = [keys[i] for i in sort_idx]
        vals_sorted = [vals[i] for i in sort_idx]

        x_ax = np.arange(len(vals))
                
        fig, ax = plt.subplots(1, 1)
        ax.bar(x_ax, vals_sorted, color='black')
        ax.set_yscale(yscale)
        
        ax.set_ylabel("Total time spend", fontsize=label_fontsize)
        ax.set_xticks(x_ax)
        ax.set_xticklabels(keys_sorted, rotation = 90)
        
        plt.subplots_adjust(bottom=0.42)


log = Log()

#%%
"""
File Writing
"""
from os import listdir

@log
def get_names(cwd, raw=False):
    file_names = listdir(cwd + "/SavedOceanModels")
    if raw:
        return file_names
    files_in_folder = [file_name.split("_")[0] for file_name in file_names ]
    return unique(files_in_folder)

@log
def save(oceanModel, name, cwd):
    if name.find("_") != -1:
        raise Exception(f"name {name} can't contain the character '_'")
    
    if name in get_names(cwd):
        raise Exception(f"name {name} already exists")
    
    
    # Seabed
    seabed_h, seabed_v = oceanModel.seabed_spline.coordinates()
    
    # Thermocline
    thermo_h, thermo_v = oceanModel.thermocline_spline.coordinates()
    
    np.save(cwd + "/SavedOceanModels/" + name + "_seabed_horizontal.npy", seabed_h)
    np.save(cwd + "/SavedOceanModels/" + name + "_seabed_vertical.npy", seabed_v)
    np.save(cwd + "/SavedOceanModels/" + name + "_thermo_horizontal.npy", thermo_h)
    np.save(cwd + "/SavedOceanModels/" + name + "_thermo_vertical.npy", thermo_v)

@log
def load(name, cwd):
    seabed_h = np.load(cwd + "/SavedOceanModels/" + name + "_seabed_horizontal.npy")
    seabed_v = np.load(cwd + "/SavedOceanModels/" + name + "_seabed_vertical.npy")
    thermo_h = np.load(cwd + "/SavedOceanModels/" + name + "_seabed_horizontal.npy")
    thermo_v = np.load(cwd + "/SavedOceanModels/" + name + "_seabed_vertical.npy")
    return seabed_h, seabed_v, thermo_h, thermo_v

@log
def delete(name, cwd):
    if name not in get_names(cwd):
        raise Exception(f"name {name} doesn't exists")

    if name.find("_") != -1:
        raise Exception(f"name {name} can't contain the character '_'")
    
    for extension in ["_seabed_horizontal", "_seabed_vertical", "_thermo_horizontal", "_thermo_vertical"]:
        if name + extension + ".npy" in get_names(cwd, raw=True):
            os.remove("SavedOceanModels/" + name + extension + ".npy")
        else:
            print(f"That's weird, couldn't find {name + extension}")
    print(f"Finished removing {name}")

@log
def rsf2numpy(name, nx=None, ny=None, plotting=False):
    name = add_rsf(name)
    
    return m8r.Input(name).read()

@log
def numpy2rsf(array, directory, name, d=[0.0007, 0.0007, 0.0005], o=[0, 0, 0]):
    name = add_rsf(name)
    
    '''Writes numpy array (Shot, Time, Receiver) into optimised rsf file'''
    Out = m8r.Output(directory+os.sep+name)
    
    for i, n in enumerate(array.shape[::-1]): # RSF dimoension norm is reverse of numpy
        nbr = i+1
        Out.put('n%i' % nbr, n)     # n : points in dimension
        Out.put('d%i' % nbr, d[i])  # d : stepsize
        Out.put('o%i' % nbr, o[i])  # o : coordinate origin

#    #Out.put('label2'= 'Offset')
#    #Out.put('label3'='Time')

    Out.write(array)
    
    Out.close()
    
    fix_comma_problem(directory, name) # Shitty m8r write floats with , not .

@log
def add_rsf(name):
    if (name.endswith('.rsf')):
        return name
    return name + '.rsf'

@log
def rm_rsf(name):
    return name.rstrip('.rsf')

@log
def fix_comma_problem(path, name):
    file_name = path + '/' + name
    
    with open(file_name, 'r') as f:
        
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        regex = re.search('d([0-9])=([0-9]),([0-9])', line)
        if regex == None:
            continue
        old_str = regex.string[1:-1]
        new_str = old_str.replace(',', '.')
        lines[i] = line.replace(old_str, new_str)
        
    with open(file_name, 'w') as f:
        f.writelines(lines)

@log
def write_flow(out_name, order, zshot, yshort):
    print("???write_lines???")
    return 'Flow("%s", "OceanModel.rsf", "eikonal order=%i zshot=%f yshot=%f br1=0.0005 br2=0.0005") \n' % (out_name, order, zshot, yshort)

@log
def write_lines(sources, receivers, order=2):
    print("???write_lines???")
    lines = []
    
    lines.append('from rsf.proj import * \n')
    lines.append('Fetch("OceanModel.rsf", "/home/peter/Desktop/master_project/Madagascar/OceanModel", server = "local") \n')
    
    for i, source in enumerate(sources):
        lines.append(write_flow('source%i' % i, order, source[0], source[1]))

    for i, receiver in enumerate(receivers):
        lines.append(write_flow('receiver%i' % i, order, receiver[0], receiver[1]))
    
    lines.append('End()')
    return lines

@log
def write_file(sources, receivers, order=2, verbose=True):
    print("???write_file???")
    """
    sources[i] = (vertical, horizontal)
    """
    
    file_name = "SConstruct.py"
    lines = write_lines(sources, receivers, order=2)
    
    with open(file_name, 'w') as f:
        f.writelines(lines)
    
    if verbose:
        print(f"Successfully wrote {file_name}")


#%%
"""
Utils
"""
@log
def unique(my_list):
    unique_list = []
    
    for element in my_list:
        if element not in unique_list:
            unique_list.append(element)
    return unique_list

@log
def set_tick_fontsize(ax):
    [tick.set_fontsize(tick_fontsize) for tick in ax.get_xticklabels()]
    [tick.set_fontsize(tick_fontsize) for tick in ax.get_yticklabels()]


@log
def sin(x, depth, amplitude, wave_length, offset=0):
    return depth + amplitude * np.sin(2*np.pi*(x + offset) / wave_length)

@log
def diff1d(s):
    if len(s) < 2:
        return np.zeros_like(s)
    ds = np.roll(s, -1) - s
    ds[-1] = ds[-2]
    return ds

@log
def extremum(arr, func_type="min"):
    print("???extremum???")
    if func_type == "min":
        func = np.argmin
    elif func_type == "max":
        func = np.argmax
    else:
        print("ERROR: min_or_max = '%s' is wrong" % func_type)
        
    return np.unravel_index(func(arr, axis=None), arr.shape)

@log
def remove_outliers(arr, percentile=0.99):
    print("??remove_outliers??")
    arr_sort = np.sort(arr, axis=None)
    idx = np.floor(arr_sort.size * percentile).astype(int)
    threshold = arr_sort[idx]
    mask = (arr < threshold)
    return arr * mask + (mask == 0) * arr_sort[idx]
    
@log
def diff2d(arr, axis=0):
    print("??diff2d??")
    mask = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]]).astype(float)
    mask = mask if axis == 0 else mask.T

    d = convolve(arr, mask, mode='same')
    
    if axis == 0:
        d[ 0, :] = d[1 , :]
        d[-1, :] = d[-2, :]
    else:
        d[:,  0] = d[:,  1]
        d[:, -1] = d[:, -2]
            
    return d

#%%
"""
PLOTTING
"""
@log
def plot_List(oceanModel):
    print("??plot_List??")
    if len(oceanModel.List) == 0:
        raise Exception("List is empty")
    
    

@log
#def get_thermocline_point(oceanModel, eikonal):
#    print("??get_thermocline_point??")
#    h_ax = np.arange(0, oceanModel.h_span(), dtype=int)
#    v_ax = oceanModel.thermocline_spline(h_ax)
#    t_ax = np.array([eikonal[v_i, h_i] for h_i, v_i in zip(h_ax, v_ax)])
#    
#    h_min = h_ax[np.argmin(t_ax)]
#    v_min = v_ax[np.argmin(t_ax)]
#    return h_min, v_min

@log
def plot_travel_routes(oceanModel, source_ids, receiver_ids, title=None):
    fig = plot_oceanModel(oceanModel, title)
    ax = fig.axes[0]
    
    if not oceanModel.is_initialized:
        oceanModel.initialize()
    
    for i, source_point in enumerate(oceanModel.coordinates2index(oceanModel.sources)):
        for j, receiver_point in enumerate(oceanModel.coordinates2index(oceanModel.receivers)):
            if i not in source_ids:
                continue
            if j not in receiver_ids:
                continue
            
            source = oceanModel.source_eikonal(i)
            receiver = oceanModel.receiver_eikonal(i, j)
#            midpoint = (source_point + receiver_point) * 0.5
            oceanModel.set_T(source, receiver)
            _, h_point = oceanModel.minimize(False)
#            res = oceanModel.minimize(midpoint[0])
#            h_point = res['x']
            v_point = oceanModel.seabed_spline(h_point)
            
            horizontal = [float(source_point[0]), h_point, float(receiver_point[0])]
            vertical   = [float(source_point[1]), v_point, float(receiver_point[1])]
            ax.plot(horizontal, vertical, '-', markersize=markersize, linewidth=linewidth_thin, color='gray')

    fig.tight_layout()
    return fig
                
@log
def plot_oceanModel(oceanModel, title=None, jacobian=None):
    sources = oceanModel.coordinates2index(oceanModel.sources)
    receivers = oceanModel.coordinates2index(oceanModel.receivers)
    
    fig, ax = plt.subplots(1, 1)
    
#    h_full = np.linspace(0, oceanModel.h_span())
#    v_full = oceanModel.seabed_spline(h_full)
#    h_spline = oceanModel.seabed_spline.horizontal_0
#    v_spline = oceanModel.seabed_spline(h_spline)
    h_full, v_full = oceanModel.seabed_spline.coordinate_idxs()
    h_spline, v_spline = oceanModel.seabed_spline.coordinates()
    
#    ax.imshow(oceanModel.oceanmodel, aspect='auto', cmap=cmap)
    im = ax.contourf(oceanModel.oceanmodel, cmap=cmap)
    ax.plot(sources[:, 0], sources[:, 1], 'or', label="Sources", markersize=markersize, linewidth=linewidth)
    ax.plot(receivers[:, 0], receivers[:, 1], 'ob', label="Receivers", markersize=markersize, linewidth=linewidth)
    ax.plot(h_full, v_full, '-', color='black', label="Seabed", markersize=markersize, linewidth=linewidth)
    ax.plot(h_spline, v_spline, '*k', label="Seabed points", markersize=markersize, linewidth=linewidth)
    plt.colorbar(im, ax=ax).set_label(label="Propagation speed", size=label_fontsize)

    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)
        
    if jacobian is not None:
        if len(jacobian) != len(h_spline):
            raise Exception(f"jacobian with len {len(jacobian)} should have len {len(h_spline)}")

        for h_i, v_i, j_i in zip(h_spline, v_spline, jacobian):
            ax.plot([h_i, h_i], [v_i, v_i + j_i], '-r', markersize=markersize, linewidth=linewidth)
        ax.plot([h_i, h_i], [v_i, v_i + j_i], '-r', label="Jacobian", markersize=markersize, linewidth=linewidth)

    ax.set_xlim([0, oceanModel.h_span()])
    ax.set_ylabel("Depth", fontsize=label_fontsize)
    ax.set_xlabel("Horizontal distance", fontsize=label_fontsize)
    ax.legend(fontsize=label_fontsize)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig

@log
def plot_source_receiver(oceanmodel, source, receiver, source_point, receiver_point, seabed_spline, source_label='source', receiver_label='receiver'):
#    hspan = seabed_spline.horizontal_0[-1]
#    h_ax = np.linspace(0, hspan, num=2000)
#    v_ax = seabed_spline(h_ax)
    T = source + receiver
    h_max = oceanmodel.seabed_spline.coordinates()[0].max()
    h_ax = np.linspace(0, h_max)
    v_ax = oceanmodel.seabed_spline(h_ax)
    t_ax = oceanmodel.thermocline_spline(h_ax)

    fig = plt.figure()
    fig.tight_layout()
    gs = gridspec.GridSpec(2, 2)
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.yaxis.set_visible(False)
    fig.subplots_adjust(wspace=0)
    
    im0 = ax0.contourf(T, cmap=cmap, vmin=T.min(), vmax=T.max(), levels=np.linspace(T.min(), T.max(), 25))
    ax0.plot(source_point[0], source_point[1], '*r', label=source_label, markersize=markersize, linewidth=linewidth)
    ax0.plot(receiver_point[0], receiver_point[1], '*g', label=receiver_label, markersize=markersize, linewidth=linewidth)
    ax0.plot(h_ax, v_ax, '-k', label="Seabed", markersize=markersize, linewidth=linewidth)
    ax0.plot(h_ax, t_ax, '-b', label="Thermocline", markersize=markersize, linewidth=linewidth)
    ax0.set_ylabel("Depth", fontsize=label_fontsize)
    ax0.set_xlabel("Horizontal distance", fontsize=label_fontsize)
    ax0.set_xlim([0, h_ax[-1]])
    fig.colorbar(im0, ax=ax0, orientation='vertical').set_label(label="Time of arrival", size=label_fontsize)
    
    im1 = ax1.contourf(source, cmap=cmap, vmin=0., vmax=T.max(), levels=np.linspace(source.min(), source.max(), 25))
    ax1.plot(source_point[0], source_point[1], '*r', markersize=markersize, linewidth=linewidth)
    ax1.plot(receiver_point[0], receiver_point[1], '*g', markersize=markersize, linewidth=linewidth)
    ax1.plot(h_ax, v_ax, '-k', markersize=markersize, linewidth=linewidth)
    ax1.plot(h_ax, t_ax, '-b', markersize=markersize, linewidth=linewidth)
    ax1.set_ylabel("Depth", fontsize=label_fontsize)
    ax1.set_xlabel("Horizontal distance", fontsize=label_fontsize)
    ax1.set_xlim([0, h_ax[-1]])
    
    ax2.contourf(receiver, cmap=cmap, vmin=0., vmax=T.max(), levels=np.linspace(receiver.min(), receiver.max(), 25))
    ax2.plot(source_point[0], source_point[1], '*r', markersize=markersize, linewidth=linewidth)
    ax2.plot(receiver_point[0], receiver_point[1], '*g', markersize=markersize, linewidth=linewidth)
    ax2.plot(h_ax, v_ax, '-k', markersize=markersize, linewidth=linewidth)
    ax2.plot(h_ax, t_ax, '-b', markersize=markersize, linewidth=linewidth)
    ax2.set_xlabel("Horizontal distance", fontsize=label_fontsize)
    ax2.set_xlim([0, h_ax[-1]])
    
    fig.colorbar(im1, ax=[ax1, ax2], orientation='vertical').set_label(label="Time of arrival", size=label_fontsize)
    ax0.invert_yaxis()
    ax1.invert_yaxis()
    ax2.invert_yaxis()
    set_tick_fontsize(ax0)
    set_tick_fontsize(ax1)
    set_tick_fontsize(ax2)
    ax0.legend(fontsize=label_fontsize)

    return fig

@log
def plot_time(oceanModel, source, receiver, h_source, h_receiver):
    T = source + receiver
        
    start = 0
    end = oceanModel.h_span()
    
    h_ax = np.arange(start, end, dtype=int)
    v_ax = oceanModel.seabed_spline(h_ax)
    t_ax = np.array([T[v_i, h_i] for h_i, v_i in zip(h_ax, v_ax)])
    s_ax = np.array([source[v_i, h_i] for h_i, v_i in zip(h_ax, v_ax)])
    r_ax = np.array([receiver[v_i, h_i] for h_i, v_i in zip(h_ax, v_ax)])
    
    h_min = h_ax[np.argmin(t_ax)]
    t_min = np.min(t_ax)
    
    h_start = int(len(h_ax) * 0.25) + 1
    h_end = int(len(h_ax) * 0.75)   
    
    t_start = np.min(t_ax[h_start:h_end]) * 0.998
    t_end = np.max(t_ax[h_start:h_end])


    largest_endpoint = max(t_ax[h_start], t_ax[h_end])
    largest_value = max(t_ax[h_start:h_end+1])
    if largest_endpoint == largest_value:
        t_end -= (t_end - t_start) * 0.1
    
    fig, (ax, ax_sr) = plt.subplots(2, 1, sharex=True)
    ax.plot(h_ax, t_ax, '-k', markersize=markersize, linewidth=linewidth, label="TOA at Seabed")
    ax.plot(h_min, t_min, '*r', markersize=markersize, linewidth=linewidth, label="Shortest travel time")
    ax.set_ylabel("Propagation time", fontsize=label_fontsize)
    ax.axvline(h_source, color='red')
    ax.axvline(h_receiver, color='blue')
    ax.set_xlim([start, end])
    ax.legend(loc=1, fontsize=label_fontsize)

    ax_sr.plot(h_ax, s_ax, ':r', markersize=markersize, linewidth=linewidth, label="Source")
    ax_sr.plot(h_ax, r_ax, ':b', markersize=markersize, linewidth=linewidth, label="Receiver")
    ax_sr.set_xlabel("Horizontal distance", fontsize=label_fontsize)
    ax_sr.set_ylabel("Propagation time", fontsize=label_fontsize)
    ax_sr.axvline(h_source, color='red', label="Source position")
    ax_sr.axvline(h_receiver, color='blue', label="Receiver position")
    ax_sr.legend(fontsize=label_fontsize)
    ax_sr.set_xlim([start, end])
    
    ax_inset = inset_axes(ax, width="50%", height="50%", loc=9, borderpad=1)
    ax_inset.plot(h_ax, t_ax, '-k', markersize=markersize, linewidth=linewidth)
    ax_inset.plot(h_min, t_min, '*r', markersize=markersize, linewidth=linewidth)
    ax_inset.set_ylim([t_start, t_end])
    ax_inset.set_xlim([h_start, h_end])
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
            
    plt.subplots_adjust(wspace=0, hspace=0)
    return fig

#@log
#def plot_time(oceanModel, source, receiver, h_start, h_end):
#    T = source + receiver
#    start = 0
#    end = oceanModel.h_span() + 1
#    
#    h_ax = np.arange(start, end, dtype=int)
#    v_ax = oceanModel.seabed_spline(h_ax)
#    t_ax = np.array([T[v_i, h_i] for h_i, v_i in zip(h_ax, v_ax)])
#    s_ax = np.array([source[v_i, h_i] for h_i, v_i in zip(h_ax, v_ax)])
#    r_ax = np.array([receiver[v_i, h_i] for h_i, v_i in zip(h_ax, v_ax)])
#    
#    h_min = h_ax[np.argmin(t_ax)]
#    t_min = np.min(t_ax)
#    
#    t_start = np.min(t_ax[h_start:h_end]) * 0.99
#    t_end = np.max(t_ax[h_start:h_end]) * 1.0
##    t_max = np.max(t_ax)
##    dt = (t_max - t_min) * 0.08
##    t_lim = [t_min - dt*0.1, t_min + dt]
#    
#    fig, (ax, ax_sr) = plt.subplots(2, 1, sharex=True)
#    ax.plot(h_ax, t_ax, '-k', markersize=markersize, linewidth=linewidth, label="TOA at Seabed")
#    ax.plot(h_min, t_min, '*r', markersize=markersize, linewidth=linewidth, label="Shortest travel time")
#    ax.set_ylabel("Propagation time", fontsize=label_fontsize)
#
#    ax_sr.plot(h_ax, s_ax, ':r', markersize=markersize, linewidth=linewidth, label="Source")
#    ax_sr.plot(h_ax, r_ax, ':b', markersize=markersize, linewidth=linewidth, label="Receiver")
#    ax_sr.set_xlabel("Horizontal distance", fontsize=label_fontsize)
#    ax_sr.set_ylabel("Propagation time", fontsize=label_fontsize)
#    ax_sr.legend(fontsize=label_fontsize)
#    
#    ax_inset = inset_axes(ax, width="30%", height="50%", loc=9, borderpad=1)
#    ax_inset.plot(h_ax, t_ax, '-k', markersize=markersize, linewidth=linewidth)
#    ax_inset.plot(h_min, t_min, '*r', markersize=markersize, linewidth=linewidth)
#    ax_inset.set_ylim([t_start, t_end])
#    ax_inset.set_xlim([h_start, h_end])
#    ax_inset.set_xticks([])
#    ax_inset.set_yticks([])
#        
#    ax.legend(fontsize=label_fontsize)
#    
#    plt.subplots_adjust(wspace=0, hspace=0)
#    return fig

def plot_optimization(oceanmodel, title=None):
    if len(oceanmodel.List) == 0:
        if not oceanmodel.save_optimization:
            raise Exception("save_optimization is set to False")
        raise Exception("No optimization found!")
    
    fig, (axh, axt)  = plt.subplots(2, 1, sharex=True)
    
    for l in oceanmodel.List:
        axh.clear()
        axt.clear()
        h, t = l
        axh.plot(h, '.:', markersize=markersize, linewidth=linewidth)
        axt.plot(t, '.:', markersize=markersize, linewidth=linewidth)
        fig.suptitle(title, fontsize=title_fontsize)
        plt.pause(0.3)
    
def plot_TOA(oceanmodel):
    toas = oceanmodel.TOA()
    toas = toas.reshape([len(oceanmodel.sources), len(oceanmodel.receivers)])
    r_ax = np.arange(len(toas)) + 1
    
    fig, ax = plt.subplots(1, 1)
    for i, toa in enumerate(toas):
        ax.plot(r_ax, toa, 'o-', label=f"Receiver {i+1}", markersize=markersize, linewidth=linewidth)
    ax.set_xlabel("Source nbr", fontsize=label_fontsize)
    ax.set_ylabel("(T)ime (O)f (A)rrival", fontsize=label_fontsize)
    ax.set_title("Data produced by forward model", fontsize=title_fontsize)
    ax.set_xticks(r_ax)
    ax.legend(fontsize=label_fontsize)
    return fig
    

@log
def create_hist(arr, bins=20):
    arr = arr.flatten()
    fig, ax = plt.subplots(1, 1)
    y, x = np.histogram(arr, bins=bins)
    ax.plot((x[1:] + x[:-1])/2, y, markersize=markersize, linewidth=linewidth)
    return fig

#%%
"""
THERMOCLINE SPLINE
"""
#class ThermoclineSpline:
#    
##    def __init__(self, h_span, depth=250., amplitude=15., wave_length=100., time_offset=0.):
#    def __init__(self, h_span, v_coords):
#        self.h_ax = np.arange(h_span)
#        self.spline = PchipInterpolator(self.h_ax, s)
##        self.depth = depth
##        self.amplitude = amplitude
##        self.wave_length = wave_length
##        self.time_offset = time_offset
##        self.build_spline(0.)
#        
#    def build_spline(self, time_source=0.):
#        return
##        s = sin(self.h_ax, self.depth, self.amplitude, self.wave_length, offset=time_source + self.time_offset)
##        self.spline = PchipInterpolator(self.h_ax, s)
#
#    @log
#    def thermocline_indices(self, as_float=False):
#        v = np.round(self.spline(self.h_ax)).astype(int)
#        h = np.round(self.h_ax).astype(int)
#        return h, v
#     
#    @log
#    def __call__(self, h, as_idx=True):
#        v = self.spline(h)
#        if as_idx:
#            return np.round(v).astype(int)
#        return v

#%%
"""
SEABED SPLINE
"""   

class Spline:
    
    @log
    def __init__(self, item, vmax):
        """
        item is either a velocity profile (m x n) 
        or a tuple (horizontal axis, vertical axis)
        """

        if isinstance(item, np.ndarray):
            uniques = np.unique(item)
            if len(uniques) == 1:
                print(f"Warning: setting spline using uniform velocity field with speed {uniques[0]}")
            horizontal_0, vertical_0 = seabed_indices(item)
        elif isinstance(item, tuple):
            horizontal_0, vertical_0 = item
        
        too_large = vertical_0 > vmax-1
        vertical_0 = vertical_0 * ~too_large + too_large * (vmax-1)
        self.horizontal_0 = horizontal_0
        self.seabed_spline = PchipInterpolator(self.horizontal_0, vertical_0)
        self.vmax = vmax
        
    @log    
    def __call__(self, x, get_float=False):
        if get_float:
            return self.seabed_spline(x)
        return np.round(self.seabed_spline(x)).astype(int)

    @log
    def coordinates(self, as_float=True):
        if as_float:
            return self.horizontal_0, self.__call__(self.horizontal_0, get_float=True)
        return np.round(self.horizontal_0).astype(int), self.__call__(self.horizontal_0, get_float=False)
    
    @log
    def coordinate_idxs(self):
        x = np.arange(0, self.horizontal_0[-1].astype(int))
        y = self(x)
        return x, y

@log
def seabed_indices(arr):
    print("IS THIS EVER CALLED?!?!")
    # Highlight seabed
    d0 = diff2d(arr, axis=0)
    d1 = diff2d(arr, axis=1)
    d = np.abs(d0) + np.abs(d1)

    # Get threshold    
    y, x = np.histogram(abs(d))    
    threshold = x[int(len(x)/2)]
    
    # Get coordinates of seabed
    edge_idxs = np.argmax(d > threshold, axis=0)
    edge_idxs[0] = edge_idxs[1]
    edge_idxs[-1] = edge_idxs[-2]
    
    return np.arange(len(edge_idxs)), edge_idxs
    
    # Pick subset
#    num = int(len(edge_idxs) / 10)
#    idxs = np.linspace(0, len(edge_idxs)-1, num=num).astype(int)
#    edge_idxs = edge_idxs[idxs]
#    
#    return idxs, edge_idxs
        
#    n = 5
#    mask = np.ones(n) / n
    
#    edge_idxs = convolve(edge_idxs, mask, mode='same')
    
#    return np.arange(len(edge_idxs)), edge_idxs

#%%
"""
Inversion 
"""
class Inversion:
    
    @log
    def __init__(self, oceanModel, true_oceanModel, sigmas=None, verbose=True):#, verbose_plot_progress=True):
        # Settable variables
        self.oceanModel = oceanModel
        self.seabed_horizontal, self.m_priori = oceanModel.seabed_spline.coordinates()
        self.thermocline_horizontal = oceanModel.thermocline_spline.coordinates()[0]
#        self.cs = []
#        self.inversion_history = []
        self.get_covariance_inv(sigmas)
        # Internal variables
#        self.true_toa = None
        
        self.true_seabed = None # This is cheating!
        self.true_thermocline = None # This is also cheating!
        self.verbose = verbose
        self.set_true_data(true_oceanModel)
#        self.ratio = []
#        self.cs = []
#        self.inversion_history = []
        self.cs = [self.Cost()]
        self.inversion_history = [np.concatenate((oceanModel.seabed_spline.coordinates()[1], oceanModel.thermocline_spline.coordinates()[1]))]
        self.data = []
        
    @log
    def get_covariance_inv(self, sigmas):
        self.C_D = np.diag(sigmas ** 2)
        self.C_D_inv = np.linalg.inv(self.C_D)
        
    @log
    def get_G(self, m_i, dv, target='seabed'):
        dv_ax = np.zeros_like(m_i)
        G = np.zeros([len(self.true_toa), len(m_i)])
        for i in range(len(m_i)):
            dv_ax[i-1] = 0.
            dv_ax[i] = dv
            m_new = m_i + dv_ax
            self.set_spline(m_new, target=target)
            G[:, i] = self.get_TOA() - self.true_toa
            print(i)
        return G
    
    @log
    def derivative(self, x, dv, cost0, target='seabed'):
        # Method
        dv_ax = np.zeros_like(x)
        
        costi = np.zeros_like(x)
        
        for i in range(len(x)):
            dv_ax[i-1] = 0.
            dv_ax[i] = dv
            v_new = x + dv_ax
            self.set_spline(v_new, target=target)
            costi[i] = self.Cost()
        
        jacobian = costi - cost0
        
        return jacobian

    @log 
    def Solve(self, variation_seabed=1., variation_thermocline=1., alpha_seabed=10**6., alpha_thermo=10**5., seabed_iter=50, min_iter=2, transition_iter=4, thermocline_iter=10, plot_optimization=True, only_optimize_thermocline=False):
        """
        Does inverse optimization of intern oceanModel
        Parameters
        ----------
        variation_seabed : float 
            variation of seabed height for calcuating derivative
        variation_thermocline : float 
            variation of thermocline height for calcuating derivative
        alpha_seabed : float 
            step size for performing gradient decent on seabed points
        alpha_thermo : float 
            step size for performing gradient decent on thermocline points
        max_iter : int
            max number of iterations for seabed
        max_iter : int
            max number of iterations for seabed
        """
        # Optimization is split into seabed and thermocline epochs
        optimize_seabed = True
        transition_phase = False
        optimize_thermocline = False
        if only_optimize_thermocline:
            optimize_seabed = False
            optimize_thermocline = True
        
        if plot_optimization:
            fig_plot = plt.figure()
            gs = gridspec.GridSpec(2, 3)
            
            ax_cost = fig_plot.add_subplot(gs[:, 2])
            ax_seabed = fig_plot.add_subplot(gs[0, :2])
            ax_thermoc = fig_plot.add_subplot(gs[1, :2], sharex=ax_seabed)
            ax_seabed.xaxis.set_visible(False)
            fig_plot.subplots_adjust(hspace=0)

            axis = [ax_cost, ax_seabed, ax_thermoc]            

        # Lists for storing optimization history
        # Remember initial conditions
#        seabed_i = np.copy(self.oceanModel.seabed_spline.coordinates())[1]
        seabed_i = self.
        
        new_seabed = np.copy(seabed_i)
        thermo_i = self.oceanModel.thermocline_spline.coordinates()[1]
        new_thermo = np.copy(thermo_i)

#        self.cs = [self.Cost()]
#        self.inversion_history = [np.concatenate((seabed_i, thermo_i))]
        self.switch_idx = 0
        self.transition_iter = transition_iter
        
        self.print_message("Beginning optimization")
        i = 0
        max_iter = seabed_iter
        while i <= max_iter:
            cost = self.Cost()
            
            if cost > 5*10**4:
                break

            # Check if optimization should switch to Transition phase
            if optimize_seabed:
                switch_to_thermocline = self.switching_criteria() or i >= max_iter
                if switch_to_thermocline and i >= min_iter:
                    self.print_message("")
                    self.print_message(f"Switching to Transition after {i} iterations")
                    optimize_seabed = False
                    transition_phase = True
                    self.switch_idx = i
                    i = 0
                    max_iter = transition_iter
            
            # Check if optimization should switch to Thermocline
            if transition_phase and i >= transition_iter:
                self.print_message("")
                self.print_message(f"Switching to thermocline optimization after {i} iterations")
                transition_phase = False
                optimize_thermocline = True
                max_iter = thermocline_iter
                self.select_best_seabed()
                i = 0

            # Check if optimization should terminate
            if optimize_thermocline and self.switching_criteria(seabed_iter + transition_iter):
                self.print_message("")
                self.print_message(f"Terminating optimization after {i} iterations because of inactivity")
                self.select_best_seabed()
                break

            # Seabed optimization
            if optimize_seabed or transition_phase:
                self.print_message(f"{i} | Optimizing seabed at idx {i} of {max_iter} | Cost={cost:.5}")
                der_seabed = self.derivative(seabed_i, variation_seabed, cost, target='seabed')
                new_seabed = seabed_i - der_seabed * alpha_seabed
                self.set_spline(new_seabed, 'seabed')
                self.data.append([i, der_seabed])

            # Thermocline optimization
            if optimize_thermocline:
                break
                self.print_message(f"Optimizing thermocline at idx {i} of {max_iter}")
                der_thermo = self.derivative(thermo_i, variation_thermocline, cost, target='thermocline')
                new_thermo = thermo_i - der_thermo * alpha_thermo
                self.set_spline(new_thermo, 'thermocline')
                self.data.append([i, der_thermo])
                    
            if plot_optimization:
                self.plot_during_inversion(seabed_i, new_seabed, thermo_i, new_thermo, i, axis)
            
            # Update newest state
            seabed_i = self.oceanModel.seabed_spline.coordinates()[1]
            thermo_i = self.oceanModel.thermocline_spline.coordinates()[1]
            
            # Save and plot step
            self.cs.append(cost)
            self.inversion_history.append(np.concatenate((seabed_i, thermo_i)))

            i += 1
#                    break
        print("FINISHED!")
        if optimize_seabed:
            print(f"{i} of {max_iter} | optimize_seabed")
        if transition_phase:
            print(f"{i} of {max_iter} | transition_phase")
        if optimize_thermocline:
            print(f"{i} of {max_iter} | optimize_thermocline")
        
        self.best_idx = np.argmin(self.cs)
        self.best_c = self.cs[self.best_idx]
        self.best_model = self.inversion_history[self.best_idx]
        
        return self.best_c, self.best_model, self.best_idx, i
    
    @log
    def set_spline(self, points, target='seabed'):
        points = self.check_points(points)
        if target == 'seabed':
            self.oceanModel.set_seabed_spline((self.seabed_horizontal, points))
        elif target == 'thermocline':
            self.oceanModel.set_thermocline_spline((self.thermocline_horizontal, points))
        else:
            raise Exception ('wrong target in derivative')

    @log
    def select_best_seabed(self):
        best_idx = np.argmin(self.cs)
        nbr_seabed_points = len(self.seabed_horizontal)
        points = self.inversion_history[best_idx][:nbr_seabed_points]
        self.set_spline(points, 'seabed')
    
    @log
    def select_best_model(self):
        best_idx = np.argmin(self.cs)
        nbr_seabed_points = len(self.seabed_horizontal)
        points = self.inversion_history[best_idx][:nbr_seabed_points]
        self.set_spline(points, 'seabed')
        points = self.inversion_history[best_idx][nbr_seabed_points:]
        self.set_spline(points, 'thermocline')
    
    @log
    def switching_criteria(self, start=0):    
        if (len(self.cs[start:]) < 3):
            return False

        S = np.array([np.std(self.cs[start:i+1]) * np.sqrt(i) for i in range(start, len(self.cs))])
        dS = diff1d(S)
        if dS[-1] < 0:
            self.print_message("Switching criteria met!")
            self.print_message(dS)
        return dS[-1] < 0
    
    @log
    def Cost(self):
        test_data = self.get_TOA()
        diff = (test_data - self.true_toa).reshape([1, -1])
        return (diff @ self.C_D_inv @ diff.T)[0, 0]
#        cost = np.sum(diff ** 2)
#        sqrt_cost = np.sqrt(cost)
#        return sqrt_cost
#    
    @log
    def get_TOA(self):
        if (self.oceanModel.is_initialized):
#            self.print_message("Internal model already initialized")
            pass
        else:
#            self.print_message("Initializing internal model")
            self.oceanModel.initialize()
            
        return self.oceanModel.TOA()
    
    @log
    def m_i(self):
        return self.oceanModel.m_i
            
    @log
    def check_points(self, points):
        too_high_idx = points > self.oceanModel.v_span()
        too_low_idx = points < 0
        too_high = np.any(too_high_idx)
        too_low = np.any(too_low_idx)
        if not too_high and not too_low:
            return points
        print(f"{too_high_idx.sum()} spline points are too high, {too_low_idx.sum()} too low")
        points[too_high_idx] = self.oceanModel.v_span()
        points[too_low_idx] = 0
        return points
    
    """ Set true data and checks """
    @log
    def set_true_data(self, item):
        """
        item can be
        OceanModel
        array
        """
        if isinstance(item, OceanModel):
            self.set_data_from_OceanModel(item)
        elif isinstance(item, np.ndarray):
            self.set_data_from_array(item)
        else:
            raise Exception (f"item {item} of type {type(item)} can't create true data")
        self.check_data()
        
    @log
    def set_data_from_OceanModel(self, oceanModel):
        self.print_message("Setting true data from model")
        if not oceanModel.is_initialized:
            oceanModel.initialize()
        self.true_toa = oceanModel.TOA()
                
    @log
    def set_data_from_array(self, item):
        print("!?!!?!?!?!set_data_from_array!?!!?!?!?!")
        self.print_message("Setting true data from array")
        self.true_toa = item
        
    @log
    def check_data(self):
        if self.true_toa is None:
            raise Exception ("True TOA data is not set!")
        if not isinstance(self.true_toa, np.ndarray):
            raise Exception (f"True TOA data of type {type(self.true_toa)} is not an array!")

    @log
    def print_message(self, msg):
        if self.verbose:
            print(msg)
            
    @log
    def know_the_real_answer(self, true_oceanModel):
        self.true_seabed = true_oceanModel.seabed_spline.coordinates()[1]
        self.true_thermocline = true_oceanModel.thermocline_spline.coordinates()[1]

    def plot_reflection_points(self):
        y, x = np.histogram(self.oceanModel.List, bins=50)
        h_ax = self.oceanModel.dense_horizontal_ax
        v_ax_test = self.oceanModel.seabed_spline(h_ax, True)
        self.set_spline(self.true_seabed)
        v_ax_true = self.oceanModel.seabed_spline(h_ax, True)
        self.set_spline(self.inversion_history[0][:10])
        v_ax_initial = self.oceanModel.seabed_spline(h_ax, True)
        self.set_spline(self.inversion_history[-1][:10])
        
        fig = plt.figure()
        fig.subplots_adjust(hspace=0)
        gs = gridspec.GridSpec(4, 1)
        
        ax0 = fig.add_subplot(gs[0:2, 0])
        ax0.bar((x[1:] + x[:-1])/2, y, width = (x[1]-x[0])*0.9, color='black')
        ax0.xaxis.set_visible(False)
        ax0.set_ylabel("Count", fontsize=label_fontsize)
        ax0.set_title("Points of reflection on seabed", fontsize=title_fontsize)
        set_tick_fontsize(ax0)
        ax0.set_xlim([h_ax[0], h_ax[-1]])
        
        ax1 = fig.add_subplot(gs[2, 0])
        ax1.plot(h_ax, v_ax_initial, '-k', linewidth=linewidth, label="Initial seabed")
        ax1.plot(h_ax, v_ax_true, '-b', linewidth=linewidth, label="True seabed")
        ax1.plot(h_ax, v_ax_test, '-r', linewidth=linewidth, label="Recovered seabed")
        ax1.invert_yaxis()
        ax1.xaxis.set_visible(False)
        ax1.set_ylabel("Depth", fontsize=label_fontsize)
        ax1.legend(fontsize=label_fontsize)
        set_tick_fontsize(ax1)
        ax1.set_xlim([h_ax[0], h_ax[-1]])
        
        ax2 = fig.add_subplot(gs[3, 0])
        ax2.plot(h_ax, v_ax_true - v_ax_test, '-k', linewidth=linewidth, label="Error")
        ax2.plot(h_ax, np.zeros_like(h_ax), ':k', linewidth=linewidth)
        ax2.invert_yaxis()
        ax2.set_xlabel("Horizontal distance", fontsize=label_fontsize)
        ax2.set_ylabel("Depth", fontsize=label_fontsize)
        ax2.legend(fontsize=label_fontsize)
        set_tick_fontsize(ax2)
        ax2.set_xlim([h_ax[0], h_ax[-1]])
        

    def plot_during_inversion(self, seabed_i, new_seabed, thermo_i, new_thermo, i, axis):
        if len(self.cs) == 0:
            return
        
        ax_cost, ax_seabed, ax_thermo = axis

        # cost history
        ax_cost.clear() 
        ax_cost.semilogy(self.cs, '*:r')
        ax_cost.set_xlabel("Iteration step")
        ax_cost.set_ylabel(r"$Cost \rightarrow \Sigma_i (|residual_i|)$")
#        ax_cost.set_ylim([0, max(self.cs)*1.01])
        
        # seabed with initial value
        ax_seabed.clear() 
        ax_seabed.plot(self.seabed_horizontal, seabed_i, '*:b', label=f"v{i}")
        if new_seabed is not None:
            ax_seabed.plot(self.seabed_horizontal, new_seabed, '*-b', label=f"v{i+1}")
        if self.true_seabed is not None:
            ax_seabed.plot(self.seabed_horizontal, self.true_seabed, '-k', label="True seabed")
        ax_seabed.set_xlabel("Horizontal distance")
#        ax_seabed.set_ylabel("Depth")
        ax_seabed.legend(fontsize=label_fontsize)
        
        # thermocline curve
        ax_thermo.clear()
        ax_thermo.plot(self.thermocline_horizontal, thermo_i, ':r', label=f"Thermocline {i}")
        if new_thermo is not None:
            ax_thermo.plot(self.thermocline_horizontal, new_thermo, '*-r', label=f"Thermocline {i+1}")
        if self.true_thermocline is not None:
            ax_thermo.plot(self.thermocline_horizontal, self.true_thermocline, '-k', label="True, thermocline curve")
        ax_thermo.set_xlabel("Horizontal distance")
        ax_thermo.set_ylabel("Depth")
        ax_thermo.legend(fontsize=label_fontsize)
        
        plt.pause(0.01)

    def plot_cost_around_thermocline(self, true_oceanModel, a_diff=3, w_diff=100, t_diff=100, a_num=12, w_num=12, t_num=12, a_ax_=None, w_ax_=None, t_ax_=None):
        
        def get_parameter_ax(p_0, diff, num):
            a = np.linspace(0, diff, num=num+1)
            ax1 = p_0 - a[::-1]
            ax2 = p_0 + a[1:]
            return np.concatenate((ax1, ax2))
    
        errors = []
        
        a_0, w_0, t_0 = true_oceanModel.thermocline_spline.coordinates()[1]
        
        a_ax = get_parameter_ax(a_0, a_diff, a_num) if a_ax_ is None else a_ax_
        w_ax = get_parameter_ax(w_0, w_diff, w_num) if w_ax_ is None else w_ax_
        
        cost = np.zeros([len(a_ax), len(w_ax)])
        
        fig = None
        for a_idx, a_i in enumerate(a_ax):
            for w_idx, w_i in enumerate(w_ax):
                try:
                    state_i = np.array([a_i, w_i, t_0])
                    self.oceanModel.set_thermocline_spline(state_i)
                    
                    cost_i = self.Cost()
                    cost[a_idx, w_idx] = cost_i
                    
                    print(f"{str((a_idx * len(w_ax) + w_idx + 1) / (len(a_ax) * len(w_ax))*100)[:4]}% | Cost = {cost_i:.4} | (a, w) = ({a_i}, {w_i})")
                except Exception as e:
                    errors.append([a_idx, a_i, w_idx, w_i])
                    print(f"{str((a_idx * len(w_ax) + w_idx + 1) / (len(a_ax) * len(w_ax))*100)[:4]}% | FAILED | ({a_idx}, {w_idx})")
                    print(e)
    
            fig = self.plot_cost_field(cost, a_ax, w_ax, a_0, w_0, fig=fig)
        
        return cost, errors, fig, a_ax, w_ax

    def plot_cost_field(self, cost, a_ax, w_ax, a_0=None, w_0=None, fig=None):
        if fig is None:
            fig, ax = plt.subplots(1, 1)
        fig.clear()
        fig.add_subplot()
        ax = fig.get_axes()[0]
        j = cost[cost != 0]
        im = ax.imshow(cost, extent=[w_ax[0], w_ax[-1], a_ax[-1], a_ax[0]], aspect='auto', vmin=j.min(), vmax=j.max())
        ax.set_xlabel('Wave number')
        ax.set_ylabel('Amplitude')
        ax.plot([w_0], [a_0], '*r', label='True value')
        plt.colorbar(im)
        plt.pause(0.01)
        return fig

    @log
    def plot_switching_criteria(self):
        
        def get_S(cs):
            return np.array([np.std(cs[:i+1]) * np.sqrt(i) for i in range(len(cs))])
        
        def get_dS(cs):
            S = get_S(cs)
            return diff1d(S)
        
        n = self.switch_idx + self.transition_iter
#        cs = np.array(self.cs)
        cs1 = np.array(self.cs)[:n]
        cs2 = np.array(self.cs)[n:]
#        S = get_S(cs)
        S1 = get_S(cs1)
        S2 = get_S(cs2)
#        dS = get_dS(cs)
        dS1 = get_dS(cs1)
        dS2 = get_dS(cs2)
#        iter_ax = np.arange(len(cs))
        iter_ax1 = np.arange(1, n+1)
        iter_ax2 = np.arange(n, len(self.cs))
#        ok = dS >= 0
        ok1 = dS1 >= 0
        ok2 = dS2 >= 0
#        switch = np.argmin(ok[5:]) + 4
        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True)
        fig.tight_layout()
        fig.subplots_adjust(hspace=0)   
        
        ax0.plot(iter_ax1, cs1, '-r', linewidth=linewidth)
        ax0.plot(iter_ax2, cs2, '-r', linewidth=linewidth)
        ax0.plot(iter_ax1[ok1], cs1[ok1], '-g', linewidth=linewidth)
        ax0.plot(iter_ax2[ok2], cs2[ok2], '-g', linewidth=linewidth)
        ax0.set_ylabel("Error: E(i)",  fontsize=label_fontsize)
        ax0.axvline(n, linewidth=linewidth/2, color='black', label='Point of switch')
#        ax0.set_xlim([0, iter_ax.max()])
        ax1.plot(iter_ax1, S1, '-r', linewidth=linewidth)
        ax1.plot(iter_ax2, S2, '-r', linewidth=linewidth)
        ax1.plot(iter_ax1[ok1], S1[ok1], '-g', linewidth=linewidth)
        ax1.plot(iter_ax2[ok2], S2[ok2], '-g', linewidth=linewidth)
        ax1.set_ylabel("S(i)",  fontsize=label_fontsize)
        ax1.axvline(n, linewidth=linewidth/2, color='black')
#        ax1.set_xlim([0, iter_ax.max()])
        ax2.plot(iter_ax1, dS1, '*r', linewidth=linewidth)
        ax2.plot(iter_ax2, dS2, '*r', linewidth=linewidth)
        ax2.plot(iter_ax1[ok1], dS1[ok1], '*g', linewidth=linewidth)
        ax2.plot(iter_ax2[ok2], dS2[ok2], '*g', linewidth=linewidth)
        ax2.set_ylabel(r"$\frac{dS}{di}$",  fontsize=title_fontsize)
        ax2.set_xlabel("Iterations (i)", fontsize=label_fontsize)
        ax2.axvline(n, linewidth=linewidth/2, color='black')
        ax2.axhline(0, linewidth=linewidth/2, color='black', ls='--')
#        ax2.set_xlim([0, iter_ax.max()])
        ax0.set_title("Determining switching criteria during optimization", fontsize=title_fontsize)
        fig.legend(fontsize=label_fontsize)
        return fig

    @log
    def plot_inversion_history(self):
        if len(self.cs) == 0:
            raise Exception("No inversion history found!")
            
        n = len(self.seabed_horizontal)
        switch = self.switch_idx + self.transition_iter
        
        data = np.array(self.inversion_history)
        seabeds = data[:switch, :n]
        thermos = data[switch:, n:]
        cs = np.array(self.cs)
        seabeds_ax = np.arange(0, seabeds.shape[0])
        thermos_ax = np.arange(seabeds.shape[0], data.shape[0])
        any_thermo = thermos.shape[0] > 0
        
        gs = gridspec.GridSpec(nrows=4, ncols=2)
        fig = plt.figure()
        ax0 = fig.add_subplot(gs[0:3, 0])
        ax1 = fig.add_subplot(gs[0:3, 1])
        ax2 = fig.add_subplot(gs[3, 0])
        ax3 = fig.add_subplot(gs[3, 1])
        ax0.xaxis.set_visible(False)
        ax1.xaxis.set_visible(False)
        fig.subplots_adjust(hspace=0)
        
        for i, c in enumerate(colors[:seabeds.shape[1]]):
            ax0.plot(seabeds_ax, seabeds[:, i], color=c, linewidth=linewidth)
        ax0.set_ylabel("Depth", fontsize=label_fontsize)
        for j, (s, c) in enumerate(zip(self.true_seabed, colors[:seabeds.shape[1]])):
            ax0.plot([seabeds_ax[-1], seabeds_ax[-1]+5], [seabeds[-1, j], s], '--', color=c, label=f"point {j}", linewidth=linewidth)
        ax0.set_title("Seabed optimization", fontsize=title_fontsize)
        
        if any_thermo:
            for i, c in enumerate(colors[:thermos.shape[1]]):
                ax1.plot(thermos_ax, thermos[:, i], color=c, linewidth=linewidth)
            ax1.set_ylabel("Depth", fontsize=label_fontsize)
            for j, (t, c) in enumerate(zip(self.true_thermocline, colors[:seabeds.shape[1]])):
                ax1.plot([thermos_ax[-1], thermos_ax[-1]+5], [thermos[-1, j], t], '--', color=c, label=f"point {j}", linewidth=linewidth)
            ax1.set_title("Thermocline optimization", fontsize=title_fontsize)
        
        ax2.plot(seabeds_ax, cs[seabeds_ax], linewidth=linewidth, color='black')
        ax2.set_ylim([0, cs[seabeds_ax].max()*1.05])
        ax2.set_ylabel("Misfit", fontsize=label_fontsize)
        ax2.set_xlabel("Iteration", fontsize=label_fontsize)
        
        if any_thermo:
            ax3.plot(thermos_ax, cs[thermos_ax], linewidth=linewidth, color='black')
            ax3.set_ylim([0, cs[thermos_ax].max()*1.05])
            ax3.set_ylabel("Misfit", fontsize=label_fontsize)
            ax3.set_xlabel("Iteration", fontsize=label_fontsize)
        
        ax0.legend(fontsize=label_fontsize, loc=2)
        ax1.legend(fontsize=label_fontsize, loc=2)
        
        set_tick_fontsize(ax0)
        set_tick_fontsize(ax1)
        set_tick_fontsize(ax2)
        set_tick_fontsize(ax3)
        
        return fig

    def plot_inversion_history_2(self):
        if len(self.cs) == 0:
            raise Exception("No inversion history found!")
       
        # Get seabed and thermo coordinates
        h_seabed = self.seabed_horizontal
        h_thermo = self.thermocline_horizontal
        best_idx = np.argmin(self.cs)
        
        gs = gridspec.GridSpec(nrows=2, ncols=4)
        fig = plt.figure()
        ax0 = fig.add_subplot(gs[0, 0:3])
        ax1 = fig.add_subplot(gs[1, 0:3], sharex=ax0)
        ax0.xaxis.set_visible(False)
        ax2 = fig.add_subplot(gs[0, 3])
        ax3 = fig.add_subplot(gs[1, 3], sharex=ax2)
        ax2.xaxis.set_visible(False)
        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
        fig.subplots_adjust(wspace=0)

        # Plot seabed history
        history_seabed = self.inversion_history[:self.switch_idx]
        labeled_idxs = [int(len(history_seabed)*0.1), len(history_seabed)-1]
        for i, seabed_spline in enumerate(history_seabed):
            alpha = i / len(history_seabed)
            if i == labeled_idxs[0]:
                ax0.plot(h_seabed, seabed_spline[:len(h_seabed)], '-k', alpha=alpha, linewidth=linewidth_thin, label="Early iteration")
            if i == labeled_idxs[1]:
                ax0.plot(h_seabed, seabed_spline[:len(h_seabed)], '-k', alpha=alpha, linewidth=linewidth_thin, label="Late iteration")
            else:
                ax0.plot(h_seabed, seabed_spline[:len(h_seabed)], '-k', alpha=alpha, linewidth=linewidth_thin)
        ax0.plot(h_seabed, self.inversion_history[0][:len(h_seabed)], '-', color='gray', linewidth=linewidth_thin, label="First iteration")
        
        # Plot true and estimated seabed
        if self.true_seabed is not None:
            ax0.plot(h_seabed, self.true_seabed, '-r', linewidth=linewidth_thin, label="True model")
        ax0.plot(h_seabed, self.inversion_history[best_idx][:len(h_seabed)], '-b', linewidth=linewidth_thin, label="Recovered seabed")
        
        # Plot thermo history
        history_thermocline = self.inversion_history[self.switch_idx:]
        labeled_idxs = [int(len(history_thermocline)*0.1), len(history_thermocline)-1]
        for i, thermocline_spline in enumerate(history_thermocline):
            alpha = i / len(history_thermocline)
            if i == labeled_idxs[0]:
                ax1.plot(h_thermo, thermocline_spline[len(h_thermo):], '-k', alpha=alpha, linewidth=linewidth_thin, label="Early iteration")
            elif i == labeled_idxs[1]:
                ax1.plot(h_thermo, thermocline_spline[len(h_thermo):], '-k', alpha=alpha, linewidth=linewidth_thin, label="Late iteration")
            else:
                ax1.plot(h_thermo, thermocline_spline[len(h_thermo):], '-k', alpha=alpha, linewidth=linewidth_thin)
        ax1.plot(h_thermo, self.inversion_history[0][len(h_thermo):], '-', color='gray', linewidth=linewidth_thin, label="First iteration")
        
        # Plot true and estimated thermocline
        if self.true_thermocline is not None:
            ax1.plot(h_thermo, self.true_thermocline, '-r', linewidth=linewidth_thin, label="True model")
        ax1.plot(h_thermo, self.inversion_history[best_idx][len(h_seabed):], '-b', linewidth=linewidth_thin, label="Recovered thermocline")
        
        ax1.set_xlabel("Horizontal distance", fontsize=label_fontsize)
        ax3.set_xlabel("Horizontal distance", fontsize=label_fontsize)
        ax0.set_ylabel("Depth of Ocean floor", fontsize=label_fontsize)
        ax1.set_ylabel("Depth of Thermocline", fontsize=label_fontsize)
        
        # Plot seabed error
        if self.true_seabed is not None:
            diff = self.true_seabed- self.inversion_history[best_idx][:len(h_seabed)]
            ax2.plot(h_seabed, diff, '-r', linewidth=linewidth_thin, label="Error")
        
        # Plot thermocline error
        if self.true_thermocline is not None:
            diff = self.true_thermocline - self.inversion_history[best_idx][len(h_seabed):]
            ax3.plot(h_thermo, diff, '-r', linewidth=linewidth_thin, label="Error")
        
        ax0.legend(fontsize=label_fontsize)
        ax1.legend(fontsize=label_fontsize)
        ax2.legend(fontsize=label_fontsize)
        ax3.legend(fontsize=label_fontsize)
        
        set_tick_fontsize(ax0)
        set_tick_fontsize(ax1)
        set_tick_fontsize(ax2)
        set_tick_fontsize(ax3)
        
        return fig

            
class OceanModel:
    """
    Contains: 
        oceanmodel, 
        sources,
        receivers,
        seabed spline
    Can Do: 
        Write SConstruct,
        Run SConstruct,
        Retreive source eikonals,
        Retreive receiver eikonals
        calculate TOA of signal
    """
    
    @log
    def __init__(self, oceanmodel, sources, receivers, times, cwd, hv_seabed_points=None, hv_thermocline_points=None, step_sizes=[0.0007, 0.0007, 0.0007], save_optimization=False, speed_above=1500., speed_below=2000., verbose=True, method='Nelder-Mead'):
        # Settable variables
        self.oceanmodel = np.copy(oceanmodel)
        self.sources = sources
        self.receivers = receivers
        self.times = times
        self.cwd = cwd
        self.step_sizes = step_sizes
        self.save_optimization = save_optimization
        self.verbose = verbose
        self.set_seabed_spline(hv_seabed_points)
        self.set_thermocline_spline(hv_thermocline_points)
        self.method = method
        self.speed_above = speed_above
        self.speed_below = speed_below
        self.dense_horizontal_ax = np.arange(0, oceanmodel.shape[1])
#        self.thermocline_spline = ThermoclineSpline(self.oceanmodel.shape[1], depth=thermo_depth, amplitude=thermo_amplitude, wave_length=thermo_wave_length, time_offset=thermo_time)
        # Internal variables        
        self.T = None
        self.order = 2
#        self.h_list = []
#        self.t_list = []
        self.List = []
        self.is_initialized = False
        self.options = {'disp': self.verbose, 'xatol' : 0.1, 'fatol' : 10**-7}#, 'eps':.5}
        self.bounds = [(0, self.h_span())]

    @log
    def initialize(self):
        self.write_OceanModelRSF()
        self.write_SConstruck()
        self.try_scons()
        self.is_initialized = True
        
    @log
    def set_seabed_spline(self, hv_points):
        if hv_points is None:
            self.seabed_spline = Spline(self.oceanmodel, vmax=self.oceanmodel.shape[0])
        else:
            self.seabed_spline = Spline(hv_points, vmax=self.oceanmodel.shape[0])
#        self.is_initialized = False

    @log
    def set_thermocline_spline(self, hv_points): # OceanModel
        if hv_points is None:
            self.thermocline_spline = Spline(self.oceanmodel, vmax=self.oceanmodel.shape[0])
        else:
            self.thermocline_spline = Spline(hv_points, vmax=self.oceanmodel.shape[0])
        self.is_initialized = False

    @log
    def set_velocity(self):
        h_ax, v_ax = self.thermocline_spline.coordinate_idxs() #!!!
        
        for h_i, v_i in zip(h_ax, v_ax):
            self.oceanmodel[:v_i, h_i] = self.speed_above
            self.oceanmodel[v_i:, h_i] = self.speed_below
        self.is_initialized = False

    @log
    def TOA(self):
        """
        <--len(receivers)-->
        
        | ...receiver 1... |      ^
        | ...receiver 2... |      |
        | ...receiver 3... | len(sources)
        | ...receiver 4... |      |
        |       ...        |      v
        
        returns:
        [T(s1, r1), T(s1, r2), ..., T(s2, r1), T(s2, r2), ..., T(sn, rm)]
        """
        source_ax = self.coordinates2index(self.sources)
        receiver_ax = self.coordinates2index(self.receivers)
        n = len(source_ax)
#        h_ax = self.dense_horizontal_ax
        
        results = np.zeros(n * len(receiver_ax))
        
                
        for i, source_poifnt in enumerate(source_ax):
            for j, receiver_point in enumerate(receiver_ax):
                
                self.set_T(self.source_eikonal(i), self.receiver_eikonal(i, j))
                
#                v_ax = self.seabed_spline(h_ax)
#                t_ax = [self.T[v_i, h_i] for h_i, v_i in zip(h_ax, v_ax)]
#                results[i*n + j] = min(t_ax)
                results[i*n + j] = self.minimize()

        self.reset_T()
        
        return results
    
    def minimize(self, only_time=True):
        v_ax = self.seabed_spline(self.dense_horizontal_ax)
        t_ax = [self.T[v_i, h_i] for h_i, v_i in zip(self.dense_horizontal_ax, v_ax)]

        if self.save_optimization:        
            idx = np.argmin(t_ax)
            self.List.append(self.dense_horizontal_ax[idx])
        
        if only_time:
            return min(t_ax)
        
        idx = np.argmin(t_ax)
        return min(t_ax), self.dense_horizontal_ax[idx]

    def m_i(self):
        return self.seabed_spline.coordinates()[1]
    
#    @log
#    def minimize2(self, x0):
#        if self.save_optimization:
#            self.h_list = []
#            self.t_list = []
#            res = minimize(self.time_at_horizontal_idx, x0, method=self.method, bounds=self.bounds, options=self.options)
#            self.List.append([self.h_list, self.t_list])
#            self.h_list = []
#            self.t_list = []
#        else:
#            res = minimize(self.time_at_horizontal_idx, x0, method=self.method, bounds=self.bounds, options=self.options)
#        return res

    @log
    def write_OceanModelRSF(self):
        for i, t in enumerate(self.times):
            self.set_source_time(t)
            numpy2rsf(self.oceanmodel, self.cwd, f'OceanModel{i}')
        self.print_message("Successfully wrote OceanModel.rsf files")

    @log
    def set_source_time(self, t): #!!! WHAT TO DO?!
#        self.thermocline_spline.build_spline(t) #!!!
        self.set_velocity()

#    @log
#    def get_thermocline_state(self): #!!!
#        a = self.thermocline_spline.amplitude
#        w = self.thermocline_spline.wave_length
#        t = self.thermocline_spline.time_offset
#        return np.array([a, w, t])

#    @log
#    def set_thermocline_state(self, a, w, t): #!!!
#        self.thermocline_spline.amplitude = a
#        self.thermocline_spline.wave_length = w
#        self.thermocline_spline.time_offset = t
#        self.is_initialized = False

    @log
    def write_SConstruck(self):
        file_name = "SConstruct.py"
        
        lines = ['from rsf.proj import * \n']
        
        # Fetch OceanModeli.rsf
        for i in range(len(self.sources)):
            lines.append(self.write_fetch(i))
        
        # Write sources
        for i, source in enumerate(self.sources):
            lines.append(self.write_flow(f'source{i}', i, source[0], source[1]))
            
        # Write receivers
        for i, source in enumerate(self.sources):
            for j, receiver in enumerate(self.receivers):
                lines.append(self.write_flow(f'receiver{i}{j}', i, receiver[0], receiver[1]))
        
        lines.append("End()")
        
        with open(file_name, 'w') as f:
            f.writelines(lines)
        
        if self.verbose:
            print(f"Successfully wrote {file_name}")
        
    def write_flow(self, out_name, i, zshot, yshort):
        return f'Flow("{out_name}", "OceanModel{i}.rsf", "eikonal order={self.order} zshot={zshot} yshot={yshort} br1={self.step_sizes[0]} br2={self.step_sizes[0]}") \n'

    def write_fetch(self, i):
        return f'Fetch("OceanModel{i}.rsf", "/home/peter/Desktop/master_project/Madagascar/OceanModel", server = "local") \n'
    
    @log
    def try_scons(self):
#        res = os.system('scons')
        res = os.system('pscons')
        if res == 0:
            self.print_message("Successfully ran scons")
        else:
            raise Exception("Error with code %i" %res)

    @log
    def source_eikonal(self, i):
        return rsf2numpy(f"source{i}.rsf")
        self.print_message(f"source{i}.rsf read sucessfully!")

    @log
    def receiver_eikonal(self, i, j):
        return rsf2numpy(f"receiver{i}{j}.rsf")
        self.print_message(f"receiver{i}{j}.rsf read sucessfully!")
            
    @log
    def set_T(self, source, receiver):
        self.T = source + receiver
    
    @log
    def reset_T(self):
        self.T = None
        
    @log
    def check_T(self):
        if (self.T is None):
            raise Exception('T is not set!')
        
#    @log
#    def time_at_horizontal_idx(self, h):
#        self.check_T()
#
#        if type(h) == np.ndarray:
#            h = h[0]
#        
#        # edge conditions 
#        h_idx = np.round(h).astype(int)
#        h_idx = max(h_idx, 1)
#        h_idx = min(h_idx, self.h_span() - 1)
#        
#        # vertical position
#        v = self.seabed_spline(h_idx, get_float=True)
#        v_idx = self.seabed_spline(h_idx, get_float=False)
#       
#        # Get subset and mask
#        T = self.T[v_idx-1:v_idx+2, h_idx-1:h_idx+2]
#        
#        mask = (np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]]) * 0.0625).astype(float)
#
#        # Get dh and dv
#        dt_dv = (mask * T).sum()
#        dt_dh = (mask * T.T).sum()
#        
#        dh = h - h_idx
#        dv = v - v_idx
#        
#        dt = dh * dt_dh + dv * dt_dv
#        
#        self.h_list.append(h)
#        self.t_list.append(T[1, 1] + dt)
#                                
#        return T[1, 1] + dt

    @log
    def oceanmodel_from_spline(self, water_speed, ground_speed):
        print("!?!!?!?!?!oceanmodel_from_spline!?!!?!?!?!")
        new_velocity = np.zeros_like(self.oceanmodel)
        
        for h_i in range(self.h_span()):
            v_i = self.seabed_spline(h_i)
            new_velocity[:v_i, h_i] = water_speed
            new_velocity[v_i:, h_i] = ground_speed
        
        return new_velocity

    @log
    def index2coordinate(self, indices):
        print("!?!!?!?!?!index2coordinate!?!!?!?!?!")
        coordinates = np.array(indices, dtype=float)
        
        coordinates[:, 0] *= self.step_sizes[0]
        coordinates[:, 1] *= self.step_sizes[1]
        
        return coordinates
    
    @log
    def coordinates2index(self, coordinates):
        indices = np.array(coordinates)
        
        indices[:, 0] /= self.step_sizes[0]
        indices[:, 1] /= self.step_sizes[1]
        
        return np.round(indices).astype(int)

    @log
    def h_span(self):
        return self.oceanmodel.shape[1] -1

    @log
    def v_span(self):
        return self.oceanmodel.shape[0] -1

    @log
    def callback(self, xk):
        pass
#        if self.save_optimization:
#            self.temp_list.append(xk)
        
    @log
    def print_message(self, msg):
        if self.verbose:
            print(msg)
            
    def save(self, name):
        save(self, name, self.cwd)
    
    def load(self, name, initialize=True):
        seabed_h, seabed_v, thermo_h, thermo_v = load(name, self.cwd)
        self.set_seabed_spline((seabed_h, seabed_v))
        self.set_seabed_spline((thermo_h, thermo_v))
        if initialize:
            self.initialize()
    
    def delete(self, name):
        delete(name, self.cwd)
    
    def get_names(self):
        return get_names(self.cwd)
        
    @log
    def plot_time(self, source_ids=None, receiver_ids=None):
        source_ids = [source_ids] if type(source_ids) is int else source_ids
        receiver_ids = [receiver_ids] if type(receiver_ids) is int else receiver_ids
        source_ids = source_ids if source_ids is not None else range(len(self.sources))
        receiver_ids = receiver_ids if receiver_ids is not None else range(len(self.receivers))
        
        for i, source_idx in enumerate(self.coordinates2index(self.sources)):
            for j, receiver_idx in enumerate(self.coordinates2index(self.receivers)):
                source, receiver = self.source_eikonal(i), self.receiver_eikonal(i, j)
                
                if i not in source_ids:
                    continue
                if j not in receiver_ids:
                    continue
                return plot_time(self, source, receiver, source_idx[0], receiver_idx[0])

    @log
    def plot_source_receiver(self, source_ids=None, receiver_ids=None):
        source_ids = [source_ids] if type(source_ids) is int else source_ids
        receiver_ids = [receiver_ids] if type(receiver_ids) is int else receiver_ids
        source_ids = source_ids if source_ids is not None else range(len(self.sources))
        receiver_ids = receiver_ids if receiver_ids is not None else range(len(self.receivers))
        
        for i, source_point in enumerate(self.coordinates2index(self.sources)):
            for j, receiver_point in enumerate(self.coordinates2index(self.receivers)):
                source, receiver = self.source_eikonal(i), self.receiver_eikonal(i, j)

                if i not in source_ids:
                    continue
                if j not in receiver_ids:
                    continue
                return plot_source_receiver(self, source, receiver, source_point, receiver_point, self.seabed_spline, f"source {i}", f"receiver {j}")

    @log
    def plot_oceanmodel(self, title=None, jacobian=None):
        return plot_oceanModel(self, title, jacobian)
    
    @log
    def plot_travel_routes(self, source_ids=None, receiver_ids=None, title=None,):
        source_ids = [source_ids] if type(source_ids) is int else source_ids
        receiver_ids = [receiver_ids] if type(receiver_ids) is int else receiver_ids
        source_ids = source_ids if source_ids is not None else range(len(self.sources))
        receiver_ids = receiver_ids if receiver_ids is not None else range(len(self.receivers))
        return plot_travel_routes(self, source_ids, receiver_ids, title)
    
    @log
    def plot_optimization(self, title=None):
        return plot_optimization(self, title)
    
    def plot_List(self):
        plot_List(self)
        
    def plot_TOA(self):
        return plot_TOA(self)

if __name__=='__main__':
    print("Dude, you're running your library, dummy!")
else:
    print("Successfully loaded library")