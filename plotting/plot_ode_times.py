"""
look at the time plots for ode experiments, also have example loss curves on bottom row
"""

import numpy as np
import matplotlib.pyplot as plt

import os
import os.path as osp

import argparse

import seaborn as sns
from matplotlib.pyplot import rc
from plotting.colors_and_styles import method_colors, method_linestyles, method_markers, method_names, method_filenames


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, choices=['g1d', 'nested_spheres', 'sines10', 'sines50', 'mnist', 'cifar10', 'svhn'], default='g1d')
parser.add_argument('--batchsize', type=int, default=16)
args = parser.parse_args()


sns.set_style('whitegrid')
rc('font', family='serif')



sizes = {'g1d': np.array([5, 20, 100, 500, 1000, 1500, 2000, 2500, 3000]),
         'nested_spheres': np.array([5, 20, 100, 500, 1000, 1500, 2000, 2500, 3000]),
         'sines10': np.array([5, 20, 200, 1000, 2000]),
         'sines50': np.array([5, 20, 200, 1000, 2000]),
         'mnist': np.array([50, 200, 350, 500]),
         'cifar10': np.array([50, 200, 350, 500]),
         'svhn': np.array([50, 200, 350, 500])}

loss_sizes = {'g1d': 1000,
              'nested_spheres': 1000,
              'sines10': 2000,
              'sines50': 2000,
              'mnist': 500,
              'cifar10': 500,
              'svhn': 500}

loss_ylimits = {'g1d': np.array([-0.1, 4]),
                'nested_spheres': np.array([-0.01, 0.75]),
                'sines10': np.array([0.05, 2]),
                'sines50': np.array([0.05, 2]),
                'mnist': np.array([0.0, 1.0]),
                'cifar10': np.array([1.1, 2.0]),
                'svhn': np.array([0.3, 1.5])}

ablations = {'g1d': 'aug3',
             'nested_spheres': 'gtol0.003',
             'sines10': '10_regular',
             'sines50': '50_regular',
             'mnist': str(args.batchsize),
             'cifar10': str(args.batchsize),
             'svhn': str(args.batchsize),
             'ornstein_uhlenbeck': '10'}

nexperiments = {'g1d': 10,
                'nested_spheres': 10,
                'sines10': 5,
                'sines50': 5,
                'mnist': 3,
                'cifar10': 3,
                'svhn': 3,
                'ornstein_uhlenbeck': 5}

if args.experiment == 'mnist' or args.experiment == 'cifar10' or args.experiment == 'svhn':
    methods = ['adjoint', 'seminorm', 'gq']
else:
    methods = ['direct', 'adjoint', 'seminorm', 'gq']


markersize=5
height = 1
width = 1
axis_fontsize = 14
title_fontsize = 18
legend_fontsize = 12
legend_alpha = 0.9
metrics = ['epoch_times', 'epoch_val_metric_history']

titles = {'g1d': ['Training Time', 'MSE'],
          'nested_spheres': ['Training Time', 'Test Accuracy'],
          'sines10': ['Training Time', 'Test MSE'],
          'sines50': ['Training Time', 'Test MSE'],
          'mnist': ['Training Time', 'Test Accuracy'],
          'cifar10': ['Training Time', 'Test Accuracy'],
          'svhn': ['Training Time', 'Test Accuracy']}

"""
xaxis = {'g1d': ['Parameters', 'Hidden Width'],
         'nested_spheres': ['Parameters', 'Hidden Width'],
         'sines10': ['Parameters', 'Hidden Width'],
         'sines50': ['Parameters', 'Hidden Width'],
         'mnist': ['Parameters', 'Hidden Channels'],
         'cifar10': ['Parameters', 'Hidden Channels'],
         'svhn': ['Parameters', 'Hidden Channels']}
"""

xaxis = {'g1d': ['Hidden Width', 'Hidden Width'],
         'nested_spheres': ['Hidden Width', 'Hidden Width'],
         'sines10': ['Hidden Width', 'Hidden Width'],
         'sines50': ['Hidden Width', 'Hidden Width'],
         'mnist': ['Hidden Channels', 'Hidden Channels'],
         'cifar10': ['Hidden Channels', 'Hidden Channels'],
         'svhn': ['Hidden Channels', 'Hidden Channels']}


def add_to_plot(method, metric, x):

    label = method_names[method]
    line = method_linestyles[method]
    color = method_colors[method]
    marker = method_markers[method]
    widths = sizes[args.experiment]
    means = np.empty(len(widths))
    stds = np.empty(len(widths))
    pltx = np.empty(len(widths))
    for w in range(len(widths)):
        if args.experiment=='sines10' or args.experiment=='sines50':
            experiment = 'sines'
        else:
            experiment = args.experiment
        folder = osp.join('results', experiment, ablations[args.experiment], str(sizes[args.experiment][w]), method_filenames[method])
        to_plot = np.empty(nexperiments[args.experiment])
        for i in range(nexperiments[args.experiment]):
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

    if args.experiment=='sines10' or args.experiment=='sines50':
        experiment = 'sines'
    else:
        experiment = args.experiment
    folder = osp.join('results', experiment, ablations[args.experiment], str(loss_sizes[args.experiment]), method_filenames[method])
    
    x = np.load(osp.join(folder, str(nexperiments[args.experiment]), metric+'.npy'))
    if metric == 'epoch_times':
        x = np.cumsum(x)
    loss = np.load(osp.join(folder, str(nexperiments[args.experiment]), 'epoch_loss_history'+'.npy'))

    plt.plot(x, loss, label=label, color=color, linestyle=line)



def make_one_plot(i):
    #ax = plt.subplot(height, width, i+1)
    for m in methods:
        add_to_plot(m, metrics[i], xaxis[args.experiment][i])
    plt.xlabel(xaxis[args.experiment][i], fontsize=axis_fontsize)
    #plt.title(titles[args.experiment][i], fontsize=title_fontsize)
    if titles[args.experiment][i] == 'Training Time':
        titles[args.experiment][i] = 'Training Time (/s)'
    plt.ylabel(titles[args.experiment][i], fontsize=axis_fontsize)
    if i%3==0:
        plt.legend(fontsize=legend_fontsize, framealpha=legend_alpha, loc='upper left')  



def make_one_loss_plot(i):
    ax = plt.subplot(height, width, 1)
    if i == 3:
        metric = 'epochs'
        x_title = 'Epoch'
    else:
        metric = 'epoch_times'
        x_title = 'Wall Clock Time (/s)'
    for m in methods:
        add_loss_to_plot(m, metric)
    plt.xlabel(x_title, fontsize=axis_fontsize)
    plt.ylabel('Test Loss', fontsize=axis_fontsize)
    plt.ylim(loss_ylimits[args.experiment][0], loss_ylimits[args.experiment][1])
    if i==3:
        plt.legend(fontsize=legend_fontsize, framealpha=legend_alpha, loc='upper right') 
    if i==4:
        ax.ticklabel_format(style='sci', scilimits=(-1, 3)) 
    #plt.title('Loss vs '+x_title, fontsize=title_fontsize)




fig = plt.figure(figsize=[5*width, 3*height])
#fig.subplots_adjust(hspace=0.3, wspace=0.0)

if args.experiment == 'mnist' or args.experiment == 'cifar10' or args.experiment == 'svhn':
    name = args.experiment + str(args.batchsize)
else:
    name = args.experiment

#fig = plt.figure(figsize=[5*width, 3*height])
#make_one_plot(0)
#plt.savefig(osp.join('plotting', 'plots', name+'_times_paper.pdf'), bbox_inches='tight')


#fig = plt.figure(figsize=[5*width, 3*height])
#make_one_plot(1)
#plt.savefig(osp.join('plotting', 'plots', name+'_metric_paper.pdf'), bbox_inches='tight')


#fig = plt.figure(figsize=[5*width, 3*height])
#make_one_loss_plot(3)
#plt.savefig(osp.join('plotting', 'plots', name+'_loss_paper.pdf'), bbox_inches='tight')


#fig = plt.figure(figsize=[5*width, 3*height])
#make_one_loss_plot(4)
#plt.savefig(osp.join('plotting', 'plots', name+'_loss_clock_paper.pdf'), bbox_inches='tight')


"""
make_one_loss_plot(3)
make_one_loss_plot(4)
plt.title('Loss vs Wall Clock Time', fontsize=title_fontsize)

#plt.show()
# save figure
if args.experiment == 'mnist' or args.experiment == 'cifar10' or args.experiment == 'svhn':
    name = args.experiment + str(args.batchsize)
else:
    name = args.experiment
plt.savefig(osp.join('plotting', 'plots', name+'_times.pdf'), bbox_inches='tight')
"""


fig = plt.figure(figsize=[5*width, 4*height])
fig.subplots_adjust(hspace=0.35, wspace=0.3)
make_one_plot(0)
make_one_plot(1)

make_one_loss_plot(3)
make_one_loss_plot(4)
plt.title('Loss vs Wall Clock Time', fontsize=title_fontsize)

#plt.show()
# save figure
if args.experiment == 'mnist' or args.experiment == 'cifar10' or args.experiment == 'svhn':
    name = args.experiment + str(args.batchsize)
else:
    name = args.experiment
plt.savefig(osp.join('plotting', 'plots', name+'_times.pdf'), bbox_inches='tight')
     
        
        
        
        
        
        