"""
plotting the results of the toy gradient problem
"""

import numpy as np
import matplotlib.pyplot as plt

import os
import os.path as osp

import argparse

import seaborn as sns
from matplotlib.pyplot import rc
from plotting.colors_and_styles import method_colors, method_linestyles, method_markers, method_names


if not osp.exists(osp.join('plotting', 'plots')):
    os.makedirs(osp.join('plotting', 'plots'))


sns.set_style('whitegrid')
rc('font', family='serif')



methods = ['direct', 'adjoint', 'gq', 'mali', 'aca']


height = 2
width = 2
axis_fontsize = 14
title_fontsize = 18
legend_fontsize = 12
legend_alpha = 0.9
metrics = ['loss', 'z0', 'a', 'T']
titles = ['Relative Error in Loss', 'Relative Error in dL/dz0', 'Relative Error in dL/da', 'Relative Error in dL/dT']


method_linestyles = {'direct': '-',
                     'adjoint': '--',
                     'seminorm': '-.',
                     'gq': '-.',
                     'mali': ':',
                     'aca': (0, (3, 1, 1, 1)),
                     'sde_adjoint': '--',
                     'sde_direct': '-'}


def true_loss(z, a, T):
    return (z**2)*np.exp(2*a*T)

def dldz(z, a, T):
    return 2*z*np.exp(2*a*T)

def dlda(z, a, T):
    return 2*T*(z**2)*np.exp(2*a*T)

def dldT(z, a, T):
    return 2*a*(z**2)*np.exp(2*a*T)


def add_to_plot(method, metric):
    
    label = method_names[method]
    line = method_linestyles[method]
    color = method_colors[method]
    folder = osp.join('results/', 'toy_gradient/', method)
    t_s = np.load(osp.join(folder, 'Ts.npy'))
    loss_errors = np.load(osp.join(folder, 'loss_errors.npy'))
    a_errors = np.load(osp.join(folder, 'a_errors.npy'))
    z0_errors = np.load(osp.join(folder, 'z0_errors.npy'))
    T_errors = np.load(osp.join(folder, 'T_errors.npy'))

    loss_true = true_loss(10, 0.2, t_s)
    z0grad_true = dldz(10, 0.2, t_s)
    agrad_true = dlda(10, 0.2, t_s)
    Tgrad_true = dldT(10, 0.2, t_s)
    
    if metric == 'loss':
      to_plot = np.abs(loss_errors)/loss_true
    elif metric == 'z0':
      to_plot = np.abs(z0_errors)/z0grad_true
    elif metric == 'a':
      to_plot = np.abs(a_errors)/agrad_true
    elif metric == 'T':
      to_plot = np.abs(T_errors)/Tgrad_true
      
    downsample_n = 3
    plt.plot(t_s[::downsample_n], to_plot[::downsample_n], label=label, linestyle=line, color=color)



def make_one_plot(i):
    ax = plt.subplot(height, width, i+1)
    for m in methods:
        add_to_plot(m, metrics[i])
    plt.xlabel('Integration Time', fontsize=axis_fontsize)
    plt.ylabel('Relative Error', fontsize=axis_fontsize)
    if i%4==0:
        plt.legend(fontsize=legend_fontsize, framealpha=legend_alpha, loc='upper left')
    plt.title(titles[i], fontsize=title_fontsize)


fig = plt.figure(figsize=[6*width, 4*height])
fig.subplots_adjust(hspace=0.3, wspace=0.3)

for i in range(4):
    make_one_plot(i)


# save figure
plt.savefig(osp.join('plotting', 'plots', 'toy_gradient.pdf'), bbox_inches='tight')


