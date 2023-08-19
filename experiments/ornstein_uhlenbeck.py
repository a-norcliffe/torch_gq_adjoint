"""
Experiment on OU process
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.neural_de import neural_de
from models.modules import mlp_t, squeezedim1, identity
from models.metrics import kl_divergence_samples, sample_logp
from datasets.datasets import oudata

from torch.utils.data import DataLoader
from experiments.run_experiment import run_experiment, set_seed


parser = argparse.ArgumentParser()
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--gtol', type=float, default=0.1)
parser.add_argument('--adjoint_option', type=str,
                    choices=['adjoint_gq', 'sde_direct', 'sde_adjoint'],
                    default='adjoint_gq')
parser.add_argument('--experiment_no', type=int, default=1)
parser.add_argument('--width', type=int, default=5)
parser.add_argument('--nepochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--ncosines', type=int, default=10)
parser.add_argument('--batchsize', type=int, default=40)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--load_model', type=int, choices=[0, 1], default=0)
parser.add_argument('--save_model', type=int, choices=[0, 1], default=0)
args = parser.parse_args()


if __name__ == "__main__":

    # set the seed for consistency across runs
    set_seed(args.experiment_no)

    # set up data loaders
    data = oudata(train=True)
    batch_size = len(data) if args.batchsize == 0 else args.batchsize
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    data = oudata(train=False)
    val_loader = torch.utils.data.DataLoader(data, batch_size=len(data))

    # model
    data_dim = 1
    encoder = squeezedim1(None)
    decoder = identity()
    drift = mlp_t(data_dim, args.width)
    diffusion = mlp_t(data_dim, args.width)
    model = neural_de(drift, diffusion, encoder, decoder, backprop_option=args.adjoint_option)
    model.defunc.ncosines = args.ncosines

    # optimiser and loss function
    optimiser = optim.Adam(model.parameters(), lr=args.lr)
    loss_func = kl_divergence_samples
    val_metric = sample_logp

    # run experiment
    run_experiment(args, 'ornstein_uhlenbeck/'+str(args.ncosines), model, optimiser, train_loader, val_loader, loss_func, val_metric)



