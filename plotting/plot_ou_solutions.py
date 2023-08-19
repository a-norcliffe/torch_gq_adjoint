"""
Look at learnt SDE solutions
"""


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

import os
import os.path as osp

from models.neural_de import neural_de
from models.modules import mlp_t, squeezedim1, identity
from models.metrics import kl_divergence_samples, sample_logp
from datasets.datasets import oudata

from torch.utils.data import DataLoader
from experiments.run_experiment import run_experiment, set_seed


import seaborn as sns
from matplotlib.pyplot import rc
from plotting.colors_and_styles import colors_dict, linestyles_dict


sns.set_style('whitegrid')
rc('font', family='serif')


width = 20

data = oudata(train=False)
val_loader = torch.utils.data.DataLoader(data, batch_size=len(data))

# model
data_dim = 1
encoder = squeezedim1(None)
decoder = identity()
drift = mlp_t(data_dim, width)
diffusion = mlp_t(data_dim, width)
model = neural_de(drift, diffusion, encoder, decoder)


folder = osp.join('results', 'ornstein_uhlenbeck', '10', '20', 'adjoint_gq', '5')
model.load_state_dict(torch.load(osp.join(folder, 'trained_model.pth')))


x, t, y = data[0]

preds_sde = model.evaluate(x, t)


pred_means = torch.mean(preds_sde, dim=0).squeeze().numpy()
pred_stds = torch.std(preds_sde, dim=0).squeeze().numpy()

y_means = torch.mean(y, dim=0).squeeze().numpy()
y_stds = torch.std(y, dim=0).squeeze().numpy()

times = t[0].squeeze().numpy()


def normal(x, mu, sig):
    out = np.exp(-((x-mu)**2)/(2*sig**2))
    out *= 1/((2*np.pi*sig**2)**0.5)
    return out


# plotting part
axis_fontsize = 14
title_fontsize = 18
legend_fontsize = 12
legend_alpha = 0.9

height = 2
width = 3


def plot_normals(i, n):
    # i = time
    npoints = 100
    zero = np.zeros(npoints)
    title = str(times[i])
    true_mean = y_means[i]
    true_std = y_stds[i]

    pred_mean = pred_means[i]
    pred_std = pred_stds[i]

    ax = plt.subplot(height, width, n)
    x = np.linspace(true_mean-5*true_std, true_mean+5*true_std, npoints)
    p = normal(x, true_mean, true_std)
    plt.plot(x, p, c=colors_dict['blue'])
    ax.fill_between(x, zero, p, facecolor=colors_dict['blue'], alpha=0.3, label='True p')

    x = np.linspace(pred_mean-5*pred_std, pred_mean+5*pred_std, npoints)
    p = normal(x, pred_mean, pred_std)
    plt.plot(x, p, c=colors_dict['red'])
    ax.fill_between(x, zero, p, facecolor=colors_dict['red'], alpha=0.3, label='Pred p')

    plt.xlabel('x', fontsize=axis_fontsize)
    plt.ylabel('p(x)', fontsize=axis_fontsize)
    plt.title('t = '+title, fontsize=title_fontsize)
    plt.legend(loc='upper left', fontsize=legend_fontsize, framealpha=legend_alpha)


fig = plt.figure(figsize=[6*width, 5*height])
fig.subplots_adjust(hspace=0.3, wspace=0.2)

plot_normals(3, 1)
plot_normals(50, 2)
plot_normals(100, 3)


# now plot from different initial conditions
sns.set_style('white')
rc('font', family='serif')
linealpha = 0.2

def plot_solutions(i, n):
    x, t, y = data[i]
    title = x[0].item()
    nsolutions = 30

    preds_sde = model.evaluate(x, t)

    preds_sde = preds_sde.detach().squeeze().numpy()
    y = y.squeeze().numpy()

    times = t[0].squeeze().numpy()

    ax = plt.subplot(height, width, n)
    plt.plot(times, y[0], color=colors_dict['blue'], alpha=linealpha, label='True')
    plt.plot(times, preds_sde[0], color=colors_dict['red'], alpha=linealpha, label='Predicted')
    for j in range(1, nsolutions):
        plt.plot(times, y[j], color=colors_dict['blue'], alpha=linealpha)
        plt.plot(times, preds_sde[j], color=colors_dict['red'], alpha=linealpha)

    plt.xlabel('t', fontsize=axis_fontsize)
    plt.ylabel('x', fontsize=axis_fontsize)
    plt.title('x(0) = {:.3f}'.format(title), fontsize=title_fontsize)
    plt.legend(loc='upper left', fontsize=legend_fontsize, framealpha=legend_alpha)


plot_solutions(1, 4)
plot_solutions(13, 5)
plot_solutions(6, 6)

plt.savefig(osp.join('plotting', 'plots', 'ou_solutions.pdf'), bbox_inches='tight')