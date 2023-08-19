"""
Nested Spheres Toy Experiment, originally called g in Augmented Neural ODEs
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.neural_de import neural_de
from models.modules import mlp, mlp_t, zero_aug, linear_layer
from datasets.datasets import nested_spheres
from models.metrics import accuracy

from torch.utils.data import DataLoader
from experiments.run_experiment import run_experiment, set_seed


parser = argparse.ArgumentParser()
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--gtol', type=float, default=0.1)
parser.add_argument('--adjoint_option', type=str,
                    choices=['adjoint_gq', 'adjoint_ode','adjoint_seminorm', 'direct'], default='adjoint_gq')
parser.add_argument('--experiment_no', type=int, default=1)
parser.add_argument('--width', type=int, default=20)
parser.add_argument('--nepochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--batchsize', type=int, default=200)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--aug_dim', type=int, default=3)
parser.add_argument('--load_model', type=int, choices=[0, 1], default=0)
parser.add_argument('--save_model', type=int, choices=[0, 1], default=0)
args = parser.parse_args()


if __name__ == "__main__":

    # set the seed for consistency across runs
    set_seed(args.experiment_no)

    # set up data loaders
    data = nested_spheres(train=True)
    batch_size = len(data) if args.batchsize == 0 else args.batchsize
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    data = nested_spheres(train=False)
    val_loader = torch.utils.data.DataLoader(data, batch_size=len(data))

    # model
    data_dim = 2
    nclasses = 2
    dim = data_dim + args.aug_dim
    encoder = zero_aug(args.aug_dim)
    decoder = linear_layer(dim, nclasses, True)
    drift = mlp_t(dim, args.width)
    diffusion = None
    model = neural_de(drift, diffusion, encoder, decoder, backprop_option=args.adjoint_option)

    # optimiser and loss function
    optimiser = optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()
    val_metric = accuracy

    # run experiment
    run_experiment(args, 'nested_spheres/gtol'+str(args.gtol), model, optimiser, train_loader, val_loader, loss_func, val_metric)



