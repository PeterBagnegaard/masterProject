from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import PchipInterpolator
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
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
markersize = 15
linewidth = 5
linewidth_thin = 1

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
def sin(x, depth, amplitude, wave_length, offset=0):
    return depth + amplitude * np.sin(2*np.pi*(x + offset) / wave_length)

@log
def diff1d(s):
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
    
    fig, (ax, ax_hist, ax_diffs) = plt.subplots(1, 3)
    
    diffs = [_list[1][-1] - _list[1][0] for _list in oceanModel.List]
    ax_diffs.hist(diffs)
    ax_diffs.set_xlabel("improvement from initial value", fontsize=label_fontsize)
    ax_diffs.set_ylabel("wave_length", fontsize=label_fontsize)

    ax_hist.hist([len(l[0]) for l in oceanModel.List])
    ax_hist.set_xlabel("# Iteration steps", fontsize=label_fontsize)
    ax_hist.set_ylabel("wave_length", fontsize=label_fontsize)

    ax.set_yscale('log')
    ax.set_xlim([0, oceanModel.h_span()])
    for _list in oceanModel.List:
        h, v = _list
        ax.plot(h, v, ':', color='gray', markersize=markersize, linewidth=linewidth)
        ax.plot(h[0], v[0], '*r', markersize=markersize, linewidth=linewidth)
        ax.plot(h[-1], v[-1], '*b', markersize=markersize, linewidth=linewidth)
    n_s = len(oceanModel.sources)
    n_r = len(oceanModel.receivers)
    for i in range(n_s):
        _List = oceanModel.List[(i*n_s) : (i*n_s + n_r)]
        _list = np.array([[l_h[-1], l_v[-1]] for l_h, l_v in _List])
        ax.plot(_list[:, 0], _list[:, 1], '-k', markersize=markersize, linewidth=linewidth_thin)

@log
def get_thermocline_point(oceanModel, eikonal):
    print("??get_thermocline_point??")
    h_ax = np.arange(0, oceanModel.h_span(), dtype=int)
    v_ax = oceanModel.thermocline_spline(h_ax)
    t_ax = np.array([eikonal[v_i, h_i] for h_i, v_i in zip(h_ax, v_ax)])
    
    h_min = h_ax[np.argmin(t_ax)]
    v_min = v_ax[np.argmin(t_ax)]
    return h_min, v_min



@log
def plot_travel_routes(oceanModel, source_ids, receiver_ids, title=None):
    print("??plot_travel_routes??")
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
            midpoint = (source_point + receiver_point) * 0.5
            oceanModel.set_T(source, receiver)
            res = oceanModel.minimize(midpoint[0])
            h_point = res['x']
            v_point = oceanModel.seabed_spline(h_point)
            
            horizontal = [float(source_point[0]), h_point[0], float(receiver_point[0])]
            vertical   = [float(source_point[1]), v_point[0], float(receiver_point[1])]
            ax.plot(horizontal, vertical, '-', markersize=markersize, linewidth=linewidth_thin, color='gray')

    fig.tight_layout()
    return fig
                
@log
def plot_oceanModel(oceanModel, title=None, jacobian=None):
    print("??plot_oceanModel??")
    sources = oceanModel.coordinates2index(oceanModel.sources)
    receivers = oceanModel.coordinates2index(oceanModel.receivers)
    
    fig, ax = plt.subplots(1, 1)
    
    h_full = np.linspace(0, oceanModel.h_span())
    v_full = oceanModel.seabed_spline(h_full)
    h_spline = oceanModel.seabed_spline.horizontal_0
    v_spline = oceanModel.seabed_spline(h_spline)
    
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
    fig.tight_layout()
    return fig

@log
def plot_source_receiver(oceanmodel, source, receiver, source_point, receiver_point, seabed_spline, source_label='source', receiver_label='receiver'):
    hspan = seabed_spline.horizontal_0[-1]
    h_ax = np.linspace(0, hspan, num=2000)
    v_ax = seabed_spline(h_ax)
    T = source + receiver

    fig = plt.figure()
    fig.tight_layout()
    gs = gridspec.GridSpec(2, 2)

    ax0 = fig.add_subplot(gs[0, :])
    im0 = ax0.contourf(T, cmap=cmap, vmin=T.min(), vmax=T.max(), levels=np.linspace(T.min(), T.max(), 25))
    ax0.plot(source_point[0], source_point[1], '*r', label=source_label, markersize=markersize, linewidth=linewidth)
    ax0.plot(receiver_point[0], receiver_point[1], '*g', label=receiver_label, markersize=markersize, linewidth=linewidth)
    ax0.plot(h_ax, v_ax, '-k', label="Seabed", markersize=markersize, linewidth=linewidth)
    ax0.set_ylabel("Depth", fontsize=label_fontsize)
    ax0.set_xlabel("Horizontal distance", fontsize=label_fontsize)
    ax0.set_xlim([0, hspan])
    fig.colorbar(im0, ax=ax0, orientation='vertical').set_label(label="Time of arrival", size=label_fontsize)

    ax1 = fig.add_subplot(gs[1, 0])
    im1 = ax1.contourf(source, cmap=cmap, vmin=0., vmax=T.max(), levels=np.linspace(source.min(), source.max(), 25))
    ax1.plot(source_point[0], source_point[1], '*r', markersize=markersize, linewidth=linewidth)
    ax1.plot(receiver_point[0], receiver_point[1], '*g', markersize=markersize, linewidth=linewidth)
    ax1.plot(h_ax, v_ax, '-k', markersize=markersize, linewidth=linewidth)
    ax1.set_ylabel("Depth", fontsize=label_fontsize)
    ax1.set_xlabel("Horizontal distance", fontsize=label_fontsize)
    ax1.set_xlim([0, hspan])
    
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.contourf(receiver, cmap=cmap, vmin=0., vmax=T.max(), levels=np.linspace(receiver.min(), receiver.max(), 25))
    ax2.plot(source_point[0], source_point[1], '*r', markersize=markersize, linewidth=linewidth)
    ax2.plot(receiver_point[0], receiver_point[1], '*g', markersize=markersize, linewidth=linewidth)
    ax2.plot(h_ax, v_ax, '-k', markersize=markersize, linewidth=linewidth)
    ax2.set_xlabel("Horizontal distance", fontsize=label_fontsize)
    ax2.set_xlim([0, hspan])
    
    fig.colorbar(im1, ax=[ax1, ax2], orientation='vertical').set_label(label="Time of arrival", size=label_fontsize)
    ax0.legend()
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
#    ax_sr.legend()
#    
#    ax_inset = inset_axes(ax, width="30%", height="50%", loc=9, borderpad=1)
#    ax_inset.plot(h_ax, t_ax, '-k', markersize=markersize, linewidth=linewidth)
#    ax_inset.plot(h_min, t_min, '*r', markersize=markersize, linewidth=linewidth)
#    ax_inset.set_ylim([t_start, t_end])
#    ax_inset.set_xlim([h_start, h_end])
#    ax_inset.set_xticks([])
#    ax_inset.set_yticks([])
#        
#    ax.legend()
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
    ax.legend()
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
class ThermoclineSpline:
    
    def __init__(self, h_span, depth=250., amplitude=15., wave_length=100., time_offset=0.):
        self.h_ax = np.arange(h_span)
        self.depth = depth
        self.amplitude = amplitude
        self.wave_length = wave_length
        self.time_offset = time_offset
        self.build_spline(0.)
        
    def build_spline(self, time_source=0.):
        s = sin(self.h_ax, self.depth, self.amplitude, self.wave_length, offset=time_source + self.time_offset)
        self.spline = PchipInterpolator(self.h_ax, s)

    @log
    def thermocline_indices(self, as_float=False):
        v = np.round(self.spline(self.h_ax)).astype(int)
        h = np.round(self.h_ax).astype(int)
        return h, v
     
    @log
    def __call__(self, h, as_idx=True):
        v = self.spline(h)
        if as_idx:
            return np.round(v).astype(int)
        return v

#%%
"""
SEABED SPLINE
"""   

class SeabedSpline:
    
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
    def seabed_coordinates(self, as_float=False):
        if as_float:
            return self.horizontal_0, self.__call__(self.horizontal_0, get_float=True)
        return np.round(self.horizontal_0).astype(int), self.__call__(self.horizontal_0, get_float=False)

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
    def __init__(self, oceanModel, dv_jacobian=1., verbose=True):#, verbose_plot_progress=True):
        # Settable variables
        self.oceanModel = oceanModel
        self.s_horizontal = oceanModel.seabed_spline.horizontal_0
        self.cs = []
        self.inversion_history = []
        self.dv_jacobian = dv_jacobian
        # Internal variables
        self.true_toa = None
        self.true_seabed = None # This is cheating!
        self.true_thermocline = None # This is also cheating!
        self.verbose = verbose

    @log
    def jacobian_seabed(self, x, dv, cost0):
        # Method
        dv_ax = np.zeros_like(x)
        
        costi = np.zeros_like(x)
        
        for i in range(len(x)):
            dv_ax[i-1] = 0.
            dv_ax[i] = dv
            v_new = x + dv_ax
            self.set_seabed_spline(v_new, is_initialized=True)
            costi[i] = self.Cost()
        
        jacobian = costi - cost0
        
        return jacobian

    @log
    def jacobian_thermocline(self, state, dthermo, cost0):
        jac = np.zeros(3)
        
        # Initial state
        ai, wi, ti = state
        # Characteristic sizes
        a0, w0 = self.oceanModel.v_span()*0.1, self.oceanModel.h_span()*0.05
#        a0, w0, t0 = self.thermocline0
        
        # !!! WHY DO I MULTIPLY AND NOT DIVIDE THESE VALUES? !!!

        # amplitude
        a_step = a0 * dthermo
        self.oceanModel.set_thermocline_state(ai + a_step, wi, ti)
        jac[0] = (self.Cost() - cost0) * a_step

        # wave_length
        w_step = w0 * dthermo
        self.oceanModel.set_thermocline_state(ai, wi + w_step, ti)
        jac[1] = (self.Cost() - cost0)  * w_step

        # time (!!!THIS IS A FACTOR 10^4 BIGGER THAN THE OTHER TWO!!!)
        t_step = w0 * dthermo
        self.oceanModel.set_thermocline_state(ai, wi, ti + t_step)
        jac[2] = (self.Cost() - cost0) * t_step
        
        self.oceanModel.set_thermocline_state(ai, wi, ti)
        return jac

    @log 
    def Solve(self, dv=1., dthermo=1., alpha_seabed=10**6., alpha_thermo=10**5., max_iter=50, min_iter=2, thermocline_iter=10, plot_optimization=True):
        # Optimization is split into seabed and thermocline epochs
        optimize_seabed = True
        optimize_thermocline = False
        
        # Remember initial conditions
        h0, vi = np.copy(self.oceanModel.seabed_spline.seabed_coordinates(True))
        new_v = np.copy(vi)
        thermo_i = self.oceanModel.get_thermocline_state() # t0, w0, a0
        depth = self.oceanModel.thermocline_spline.depth
        self.thermocline0 = np.copy(thermo_i) # We use this for updating thermocline parameters

        if plot_optimization:
            fig_plot, figs = plt.subplots(3, 2)

        # Lists for storing optimization history
        self.cs = [self.Cost()]
        self.inversion_history = [np.concatenate((vi, thermo_i))]
        
#        for i in range(max_iter):
        i = 0
        while i <= max_iter:
            cost = self.Cost()

            # Check if optimization should switch
            if optimize_seabed:
                switch_to_thermocline = self.switching_criteria() or i >= max_iter
                if switch_to_thermocline and i >= min_iter:
                    optimize_seabed = False
                    optimize_thermocline = True
                    self.switch_idx = i
                    print(f"Switching after {i} iterations")
                    max_iter = thermocline_iter
                    i = 0

            # Seabed step
            if optimize_seabed:
                # !!! FORGOT TO DIVIDE BY dv !!!
                der_seabed = self.jacobian_seabed(vi, dv, cost)
                new_v = vi - der_seabed * alpha_seabed
                self.set_seabed_spline(new_v)

            # Thermocline step
            if optimize_thermocline:
                break #!!!
                der_thermo = self.jacobian_thermocline(thermo_i, dthermo, cost)
                new_t = thermo_i - der_thermo * alpha_thermo
                self.set_thermocline_spline(new_t)
        
            # Save and plot step
            self.cs.append(cost)
            self.inversion_history.append(np.concatenate((vi, thermo_i)))
            self.print_message(f"{i}: Cost={cost}")
            
            if plot_optimization:
                self.plot_during_inversion(h0, vi, new_v, i, figs, depth)
            
            # Update newest state
            _, vi = self.oceanModel.seabed_spline.seabed_coordinates(True)
            thermo_i = self.oceanModel.get_thermocline_state()
            
            i += 1
#                    break
        
        self.best_idx = np.argmin(self.cs)
        self.best_c = self.cs[self.best_idx]
        self.best_model = self.inversion_history[self.best_idx]
        
        return self.best_c, self.best_model, self.best_idx, i
        
    @log
    def switching_criteria(self):    
        if (len(self.cs) < 3):
            return False
        
        S = np.array([np.std(self.cs[:i+1]) for i in range(len(self.cs))])
        dS = diff1d(S)
        return dS[-1] < 0
        
    def plot_during_inversion(self, h0, vi, new_v, i, figs, depth):       
        ((fa, fb), (fc, fd), (fe, ff)) = figs
        amplitudes = [state_i[-3] for state_i in self.inversion_history]   
        wave_lengths = [state_i[-2] for state_i in self.inversion_history]
        times = [state_i[-1] for state_i in self.inversion_history]
        axis = np.arange(len(times))
        ones = np.ones_like(times)
        thermocline_ax = np.arange(h0.min(), h0.max())
        thermocline_curve = sin(thermocline_ax, depth, amplitudes[-1], wave_lengths[-1], times[-1])
        
        # cost history
        fa.clear() 
        fa.plot(self.cs, '*:r')
        fa.set_xlabel("Iteration step")
        fa.set_ylabel(r"$Cost \rightarrow \Sigma_i (|residual_i|)$")
        fa.set_ylim([0, max(self.cs)*1.01])
        
        # time offset
        ff.clear() 
        if self.true_thermocline is not None:
            ff.plot(axis, ones * self.true_thermocline[2], '-k', label="True time offset")
        ff.plot(times, '*:r', label="time offset")
        ff.set_xlabel("Iteration step")
        ff.legend()
        
        # seabed with initial value
        fc.clear() 
        fc.plot(h0, vi, '*:b', label=f"v{i}")
        if new_v is not None:
            fc.plot(h0, new_v, '*-b', label=f"v{i+1}")
        if self.true_seabed is not None:
            fc.plot(h0, self.true_seabed, '*:r', label="True seabed")
        fc.set_xlabel("Horizontal distance")
        fc.set_ylabel("Depth")
        fc.legend()

        # amplitude
        fb.clear() 
        if self.true_thermocline is not None:
            fb.plot(axis, ones * self.true_thermocline[0], '-k', label="True amplitude")
        fb.set_title(r'$depth + amplitude \cdot sin(wave\_number \cdot t + time\_offset)$')
        fb.plot(amplitudes, '*:r', label="amplitude")
        fb.legend()
        
        # thermocline curve
        fe.clear()
        fe.plot(thermocline_ax, thermocline_curve, ':r', label="Thermocline curve")
        if self.true_thermocline is not None:
            thermocline_curve_true = sin(thermocline_ax, depth, self.true_thermocline[0], self.true_thermocline[1], self.true_thermocline[2])
            fe.plot(thermocline_ax, thermocline_curve_true, '-', color='gray', label="True, thermocline curve")
        fe.set_xlabel("Horizontal distance")
        fe.set_ylabel("Depth")
        fe.legend()

        # wave_length
        fd.clear() 
        if self.true_thermocline is not None:
            fd.plot(axis, ones * self.true_thermocline[1], '-k', label="True wave_length")
        fd.plot(wave_lengths, '*:r', label="wave numbers")
        fd.legend()
        
        plt.pause(0.01)        

    def plot_inversion_history(self):
        if len(self.cs) == 0:
            raise Exception("No inversion history found!")
            
        fig, (ax0, ax1) = plt.subplots(1, 2)
        fig.tight_layout()
        
        h_ax = self.s_horizontal
        best_idx = np.argmin(self.cs)
        
        for i, seabed_spline in enumerate(self.inversion_history):
            alpha = i / len(self.inversion_history)
            ax0.plot(h_ax, seabed_spline[:len(h_ax)], '-k', alpha=alpha, linewidth=linewidth)
        
        if self.true_seabed is not None:
            ax0.plot(h_ax, self.true_seabed, '-r', linewidth=linewidth, label="True model")
        ax0.plot(h_ax, self.inversion_history[0][:len(h_ax)], '-k', linewidth=linewidth, label="Initial model")
        ax0.plot(h_ax, self.inversion_history[best_idx][:len(h_ax)], '-b', linewidth=linewidth, label="Recovered model")
        
        ax0.set_xlabel("Horizontal distance", fontsize=label_fontsize)
        ax0.set_ylabel("Depth", fontsize=label_fontsize)
        ax1.plot(self.cs, '*:r', label="Cost", linewidth=linewidth)
        ax1.text(30, 4*10**-5, f"Smallest cost: \n{self.cs[best_idx]:.2}", fontsize=label_fontsize)
        ax0.legend(fontsize=label_fontsize)
        ax1.set_xlabel("Iterations", fontsize=label_fontsize)
        ax1.set_ylabel("Cost", fontsize=label_fontsize)
        ax1.legend(fontsize=label_fontsize)
        fig.suptitle("Plot of inversion method", fontsize=title_fontsize)

#        if len(self.cs) == 0:
#            raise Exception("No inversion history found!")
#            
#        fig, (ax0, ax1) = plt.subplots(1, 2)
#        fig.tight_layout()
#        
#        h_ax = self.s_horizontal
#        best_idx = np.argmin(self.cs)
#        
#        for i, seabed_spline in enumerate(self.inversion_history):
#            alpha = i / len(self.inversion_history)
#            ax0.plot(h_ax, seabed_spline[:len(h_ax)], '-k', alpha=alpha, linewidth=0.5)
#        
#        ax0.plot(h_ax, self.inversion_history[0][:len(h_ax)], '-k', linewidth=1, label="Initial model")
#        ax0.plot(h_ax, self.inversion_history[best_idx][:len(h_ax)], '-b', linewidth=1, label="Recovered model")
#        if self.true_seabed is not None:
#            ax0.plot(h_ax, self.true_seabed, '-r', linewidth=1, label="True model")
#        ax0.set_xlabel("Horizontal distance")
#        ax0.set_ylabel("Depth")
#        ax1.plot(self.cs, '*:r', label="Cost")
#        ax1.text(30, 4*10**-5, f"Smallest cost: \n{self.cs[best_idx]:.2}")
#        ax0.legend()
#        ax1.set_xlabel("Iterations")
#        ax1.set_ylabel("Cost")
#        ax1.legend()
#        fig.suptitle("Plot of inversion method")
#        return fig
        
    @log
    def Cost(self):
        # !!!This shouldn't be sqrt!!!
        test_data = self.get_TOA()
        diff = test_data - self.true_toa
#        return np.sum(diff ** 2)
        return np.sqrt( np.sum(diff ** 2) )
    
    @log
    def get_TOA(self):
        if (self.oceanModel.is_initialized):
            self.print_message("Internal model already initialized")
        else:
            self.print_message("Initializing internal model")
            self.oceanModel.initialize()
            
        return self.oceanModel.TOA()
    
    @log
    def set_seabed_spline(self, points, is_initialized=False):
        points = self.check_points(points)
        self.oceanModel.set_seabed_spline((self.s_horizontal, points))
        self.oceanModel.is_initialized = is_initialized
#        self.set_velocity( self.oceanModel.oceanmodel_from_spline(self.water_speed, self.ground_speed) )
        
    def set_thermocline_spline(self, state, is_initialized=False):
        self.oceanModel.set_thermocline_state(*state)
        self.oceanModel.is_initialized = is_initialized
#        self.set_velocity( self.oceanModel.oceanmodel_from_spline(self.water_speed, self.ground_speed) )
        
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
    def know_the_real_answer(self, seabed_spline, thermocline_state):
        _, self.true_seabed = seabed_spline.seabed_coordinates(True)
        self.true_thermocline = thermocline_state

    @log
    def plot_switching_criteria(self):
        
        def get_S(cs):
            return np.array([np.std(cs[:i+1]) for i in range(len(cs))])
        
        def get_dS(cs):
            S = get_S(cs)
            return diff1d(S)
        
        fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
        fig.tight_layout()
        fig.subplots_adjust(hspace=0)   
        
        cs = np.array(self.cs)
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
    def __init__(self, oceanmodel, sources, receivers, times, cwd, hv_points=None, step_sizes=[0.0007, 0.0007, 0.0007], save_optimization=False, speed_above=1500., speed_below=2000., thermo_depth=250., thermo_amplitude=15., thermo_wave_length=100., thermo_time=0., verbose=True, method='Nelder-Mead'):
        # Settable variables
        self.oceanmodel = np.copy(oceanmodel)
        self.sources = sources
        self.receivers = receivers
        self.times = times
        self.cwd = cwd
        self.step_sizes = step_sizes
        self.save_optimization = save_optimization
        self.verbose = verbose
        self.set_seabed_spline(hv_points)
        self.method = method
        self.speed_above = speed_above
        self.speed_below = speed_below
        self.thermocline_spline = ThermoclineSpline(self.oceanmodel.shape[1], depth=thermo_depth, amplitude=thermo_amplitude, wave_length=thermo_wave_length, time_offset=thermo_time)
        # Internal variables        
        self.source_names = []
        self.receiver_names = []
        self.T = None
        self.order = 2
        self.h_list = []
        self.t_list = []
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
            self.seabed_spline = SeabedSpline(self.oceanmodel, vmax=self.oceanmodel.shape[0])
        else:
            self.seabed_spline = SeabedSpline(hv_points, vmax=self.oceanmodel.shape[0])
        self.is_initialized = False

    @log
    def set_velocity(self):
        h_ax, v_ax = self.thermocline_spline.thermocline_indices()
        
        for h_i, v_i in zip(h_ax, v_ax):
            self.oceanmodel[:v_i, h_i] = self.speed_above
            self.oceanmodel[v_i:, h_i] = self.speed_below
        self.is_initialized = False

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
        results = []
                
        for i, source_point in enumerate(self.coordinates2index(self.sources)):
            for j, receiver_point in enumerate(self.coordinates2index(self.receivers)):
                
                self.set_T(self.source_eikonal(i), self.receiver_eikonal(i, j))
                
                v_poss = (source_point[0], receiver_point[0])
                start = min(v_poss) 
                end = max(v_poss)
                midpoint = np.round((end + start) * 0.5).astype(float)
                
                h_ax = np.arange(start, end, dtype=int)
                v_ax = self.seabed_spline(h_ax)
                
                t_ax = [self.T[v_i, h_i] for h_i, v_i in zip(h_ax, v_ax)]
                
                s = PchipInterpolator(h_ax, t_ax)
                
                res = minimize(s, midpoint, method=self.method, bounds=[(start, end)], options=self.options)
        
#                results.append(res['x'])
                results.append(res['fun'])

        self.reset_T()
        
        return np.array(results)
    
    
    @log
    def minimize(self, x0):
        if self.save_optimization:
            self.h_list = []
            self.t_list = []
            res = minimize(self.time_at_horizontal_idx, x0, method=self.method, bounds=self.bounds, options=self.options)
            self.List.append([self.h_list, self.t_list])
            self.h_list = []
            self.t_list = []
        else:
            res = minimize(self.time_at_horizontal_idx, x0, method=self.method, bounds=self.bounds, options=self.options)
        return res

    @log
    def write_OceanModelRSF(self):
        for i, t in enumerate(self.times):
            self.set_source_time(t)
            numpy2rsf(self.oceanmodel, self.cwd, f'OceanModel{i}')
        self.print_message(f"Successfully wrote OceanModel.rsf files")

    @log
    def set_source_time(self, t):
        self.thermocline_spline.build_spline(t)
        self.set_velocity()

    @log
    def get_thermocline_state(self):
        a = self.thermocline_spline.amplitude
        w = self.thermocline_spline.wave_length
        t = self.thermocline_spline.time_offset
        return np.array([a, w, t])
    
    @log
    def set_thermocline_state(self, a, w, t):
        self.thermocline_spline.amplitude = a
        self.thermocline_spline.wave_length = w
        self.thermocline_spline.time_offset = t
        self.is_initialized = False

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
    def get_TOA_curve(self, source, receiver, plot=True):
        print("!?!!?!?!?!!get_TOA_curve!?!!?!?!?!!")
        self.print_message("Plotting TOA curve")
        T = source + receiver
        h, v = self.seabed_spline.seabed_coordinates()
        if plot:
            fig, ax = plt.subplots(1, 1)
            ax.plot(h, T[v, h], '+:', markersize=markersize, linewidth=linewidth)
            return fig
        else:
            return h, T[v, h]        
            
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
        
    @log
    def time_at_horizontal_idx(self, h):
        self.check_T()

        if type(h) == np.ndarray:
            h = h[0]
        
        # edge conditions 
        h_idx = np.round(h).astype(int)
        h_idx = max(h_idx, 1)
        h_idx = min(h_idx, self.h_span() - 1)
        
        # vertical position
        v = self.seabed_spline(h_idx, get_float=True)
        v_idx = self.seabed_spline(h_idx, get_float=False)
       
        # Get subset and mask
        T = self.T[v_idx-1:v_idx+2, h_idx-1:h_idx+2]
        
        mask = (np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]]) * 0.0625).astype(float)

        # Get dh and dv
        dt_dv = (mask * T).sum()
        dt_dh = (mask * T.T).sum()
        
        dh = h - h_idx
        dv = v - v_idx
        
        dt = dh * dt_dh + dv * dt_dv
        
        self.h_list.append(h)
        self.t_list.append(T[1, 1] + dt)
                                
        return T[1, 1] + dt

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
                return plot_source_receiver(self.oceanmodel, source, receiver, source_point, receiver_point, self.seabed_spline, f"source{i}", f"receiver{i}{j}")

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