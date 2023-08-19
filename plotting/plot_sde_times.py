"""
plot ou figure, and print ablation times
"""

import numpy as np
import matplotlib.pyplot as plt

import os
import os.path as osp

import argparse

import seaborn as sns
from matplotlib.pyplot import rc
from plotting.colors_and_styles import method_colors, method_linestyles, method_markers, method_names, method_filenames


if not osp.exists(osp.join('plotting', 'plots')):
    os.makedirs(osp.join('plotting', 'plots'))


sns.set_style('whitegrid')
rc('font', family='serif')



sizes = {'gq': np.array([5, 20, 50, 100]),
         'sde_adjoint': np.array([5, 20, 50, 100]),
         'sde_direct': np.array([5, 20, 50])}


methods = ['sde_direct', 'sde_adjoint', 'gq']



markersize=5
height = 1
width = 1
axis_fontsize = 14
title_fontsize = 18
legend_fontsize = 12
legend_alpha = 0.9

metrics = ['epoch_times', 'epoch_loss_history']
titles = ['Training Time', 'Test KL Divergence']
xaxis = ['Hidden Width', 'Hidden Width']


def add_to_plot(method, metric, x):

    label = method_names[method]
    line = method_linestyles[method]
    color = method_colors[method]
    marker = method_markers[method]
    widths = sizes[method]
    means = np.empty(len(widths))
    stds = np.empty(len(widths))
    pltx = np.empty(len(widths))
    for w in range(len(widths)):
        folder = osp.join('results', 'ornstein_uhlenbeck', '10', str(sizes[method][w]), method_filenames[method])
        to_plot = np.empty(5)
        for i in range(5):
            a = np.load(osp.join(folder, str(i+1), metric+'.npy'))
            if metric == 'epoch_times':
                a = np.cumsum(a)
            to_plot[i] = a[-1]
        means[w] = np.mean(to_plot)
        stds[w] = np.std(to_plot)
        if x == 'Parameters':
            a = np.load(osp.join(folder, str(i+1), 'nparams.npy'))
            #pltx[w] = a
            pltx[w] = widths[w]
        else:
            pltx[w] = widths[w]
    plt.errorbar(x=pltx, y=means, yerr=stds, label=label, color=color, linestyle=line)


def add_loss_to_plot(method, metric):
    # metric is 'epoch_times' or 'epochs'
    label = method_names[method]
    line = method_linestyles[method]
    color = method_colors[method]
    marker = method_markers[method]

    folder = osp.join('results', 'ornstein_uhlenbeck', '10', '100', method_filenames[method])
    
    x = np.load(osp.join(folder, '5', metric+'.npy'))
    if metric == 'epoch_times':
        x = np.cumsum(x)
    loss = np.load(osp.join(folder, '5', 'epoch_loss_history'+'.npy'))

    plt.plot(x, loss, label=label, color=color, linestyle=line)


def make_one_plot(i):
    #ax = plt.subplot(height, width, i+1)
    for m in methods:
        add_to_plot(m, metrics[i], xaxis[i])
    plt.xlabel(xaxis[i], fontsize=axis_fontsize)
    #plt.title(titles[i], fontsize=title_fontsize)
    if titles[i] == 'Training Time':
        titles[i] = 'Training Time (/s)'
    plt.ylabel(titles[i], fontsize=axis_fontsize)
    if i==1:
        plt.legend(fontsize=legend_fontsize, framealpha=legend_alpha, loc='upper right')
    



def make_one_loss_plot(i):
    #ax = plt.subplot(height, width, i)
    if i == 3:
        metric = 'epochs'
        x_title = 'Epoch'
    else:
        metric = 'epoch_times'
        x_title = 'Wall Clock Time (/s)'

    add_loss_to_plot('sde_adjoint', metric)
    add_loss_to_plot('gq', metric)
    plt.xlabel(x_title, fontsize=axis_fontsize)
    plt.ylabel('Test KL Divergence', fontsize=axis_fontsize)
    plt.ylim(0.04, 0.5)
    #plt.title('Loss vs '+x_title, fontsize=title_fontsize)
    if i%3==0:
        plt.legend(fontsize=legend_fontsize, framealpha=legend_alpha, loc='upper right')  



fig = plt.figure(figsize=[5*width, 3*height])
#fig.subplots_adjust(hspace=0.3, wspace=0.0)
make_one_plot(0)
plt.savefig(osp.join('plotting', 'plots', 'ou_times_paper.pdf'), bbox_inches='tight')


fig = plt.figure(figsize=[5*width, 3*height])
#fig.subplots_adjust(hspace=0.3, wspace=0.0)
make_one_plot(1)
plt.savefig(osp.join('plotting', 'plots', 'ou_metric_paper.pdf'), bbox_inches='tight')


fig = plt.figure(figsize=[5*width, 3*height])
make_one_loss_plot(3)
plt.savefig(osp.join('plotting', 'plots', 'ou_loss_paper.pdf'), bbox_inches='tight')


fig = plt.figure(figsize=[5*width, 3*height])
make_one_loss_plot(4)
plt.savefig(osp.join('plotting', 'plots', 'ou_loss_clock_paper.pdf'), bbox_inches='tight')

"""
make_one_loss_plot(3)
plt.title('Loss vs Epoch', fontsize=title_fontsize)
make_one_loss_plot(4)
plt.title('Loss vs Wall Clock Time', fontsize=title_fontsize)

# save figure
plt.savefig(osp.join('plotting', 'plots', 'ou_times.pdf'), bbox_inches='tight')
"""

"""

# now print the times taken during ablation and final kl

def get_values(ncosines):
    folder = 'results/ornstein_uhlenbeck/'+str(ncosines)+'/20/adjoint_gq'
    total_times = []
    total_mse = []
    for ex in range(1, 5+1):
        times = np.load(osp.join(folder, str(ex), 'epoch_times.npy'))
        val_metric = np.load(osp.join(folder, str(ex), 'epoch_loss_history.npy'))
        times =  np.cumsum(times)
        total_times.append(times[-1])
        total_mse.append(val_metric[-1])
    total_times = np.array(total_times)/60 #get in terms of minutes
    total_mse = np.array(total_mse)
    mean_time = np.mean(total_times)
    std_time = np.std(total_times)
    mean_mse = np.mean(total_mse)
    std_mse = np.std(total_mse)
    print(ncosines)
    print('Time: {} +- {}'.format(mean_time, std_time))
    print('KL: {} +- {}'.format(mean_mse, std_mse))
    print('\n')




get_values(0)
get_values(1)
get_values(5)
get_values(10)
get_values(25)
get_values(50)

"""