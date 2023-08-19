"""
Experiment on lots of sine curves
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.neural_de import neural_de
from models.modules import mlp_sonode, identity, remove_aug 
from datasets.datasets import sinedata

from torch.utils.data import DataLoader
from experiments.run_experiment import run_experiment, set_seed


parser = argparse.ArgumentParser()
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--gtol', type=float, default=0.1)
parser.add_argument('--adjoint_option', type=str,
                    choices=['adjoint_gq', 'adjoint_ode','adjoint_seminorm', 'direct'], default='adjoint_gq')
parser.add_argument('--experiment_no', type=int, default=1)
parser.add_argument('--regular_times', type=str, choices=['regular', 'irregular'], default='regular')
parser.add_argument('--ntimes', type=int, choices=[10, 50], default=10)
parser.add_argument('--width', type=int, default=100)
parser.add_argument('--nepochs', type=int, default=250)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batchsize', type=int, default=0)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--load_model', type=int, choices=[0, 1], default=0)
parser.add_argument('--save_model', type=int, choices=[0, 1], default=0)
args = parser.parse_args()


if __name__ == "__main__":

    # set the seed for consistency across runs
    set_seed(args.experiment_no)

    # set up data loaders
    use_reg = True if args.regular_times == 'regular' else False
    data = sinedata(train=True, ntimes=args.ntimes, regular=use_reg)
    batch_size = len(data) if args.batchsize == 0 else args.batchsize
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    data = sinedata(train=False, ntimes=args.ntimes, regular=use_reg)
    val_loader = torch.utils.data.DataLoader(data, batch_size=len(data))

    # model
    data_dim = 1
    encoder = identity()
    decoder = remove_aug(data_dim)
    drift = mlp_sonode(data_dim, args.width)
    diffusion = None
    model = neural_de(drift, diffusion, encoder, decoder, backprop_option=args.adjoint_option)

    # optimiser and loss function
    optimiser = optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.MSELoss()

    # run experiment
    run_experiment(args, 'sines/'+str(args.ntimes)+'_'+args.regular_times, model, optimiser, train_loader, val_loader, loss_func, loss_func)



