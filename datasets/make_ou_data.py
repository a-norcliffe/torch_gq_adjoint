"""
generate data from a time dependent ou process
"""

import numpy as np
import matplotlib.pyplot as plt
import sdeint
import torch
import argparse
import os
import os.path as osp

if not osp.exists('datasets/data/ornstein_uhlenbeck'):
    os.mkdir('datasets/data/ornstein_uhlenbeck')

mu = 0.2
theta = 0.1
sigma = 0.6
phi = 0.15
tstart = 0
tend = 10


def f(y, t):
    return (mu*t - theta*y)


def g(y, t):
    return sigma + phi*t


def solve_sde(y0):
    return sdeint.stratint(f, g, y0, times)


for name in ['train', 'val']:

    if name=='train':
        ninitial = 200
        nsamples = 45
        ntimes = 101
        np.random.seed(982846947) #any seed was typed in randomly
    else:
        ninitial = 20
        nsamples = 45
        ntimes = 101
        np.random.seed(535617672) #any seed was typed in randomly

    times = np.linspace(tstart, tend, ntimes)
    ys = np.empty((ninitial, nsamples, ntimes, 1))

    for i in range(ninitial):
        y0 = 6*np.random.rand()-3
        for j in range(nsamples):
            y = solve_sde(y0)
            ys[i][j] = y

    ys = torch.tensor(ys).float()

    times = torch.tensor(times).float().unsqueeze(-1)
    torch.save(ys, 'datasets/data/ornstein_uhlenbeck/ou_y_'+name+'.pt')
    torch.save(times, 'datasets/data/ornstein_uhlenbeck/ou_t_'+name+'.pt')



