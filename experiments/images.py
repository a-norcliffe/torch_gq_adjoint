"""
Image classification experiments, can do MNIST, CIFAR10, SVHN
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.neural_de import neural_de
from models.modules import downsampling, convolutions, fc_layers
from datasets.datasets import image_data
from models.metrics import accuracy

from torch.utils.data import DataLoader
from experiments.run_experiment import run_experiment, set_seed


parser = argparse.ArgumentParser()
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--gtol', type=float, default=0.1)
parser.add_argument('--adjoint_option', type=str,
                    choices=['adjoint_gq', 'adjoint_ode','adjoint_seminorm', 'direct'], default='adjoint_gq')
parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10', 'svhn'], default='mnist')
parser.add_argument('--experiment_no', type=int, default=1)
parser.add_argument('--state_width', type=int, default=10)
parser.add_argument('--width', type=int, default=50)
parser.add_argument('--nepochs', type=int, default=15)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batchsize', type=int, default=32)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--load_model', type=int, choices=[0, 1], default=0)
parser.add_argument('--save_model', type=int, choices=[0, 1], default=0)
args = parser.parse_args()


if __name__ == "__main__":

    # set the seed for consistency across runs
    set_seed(args.experiment_no)

    # select values based on the images being used
    if args.dataset == 'mnist':
        in_channels = 1
        nhidden = 50
        nclasses = 10
        val_batch = 500
    elif args.dataset == 'cifar10':
        in_channels = 3
        nhidden = 50
        nclasses = 10
        val_batch = 500
    elif args.dataset == 'svhn':
        in_channels = 3
        nhidden = 50
        nclasses = 10
        val_batch = 16

    # set up data loaders
    data = image_data(dataset=args.dataset, train=True)
    batch_size = len(data) if args.batchsize == 0 else args.batchsize
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    data = image_data(dataset=args.dataset, train=False)
    val_loader = torch.utils.data.DataLoader(data, batch_size=val_batch)
    
    # model
    nchannels = args.width
    encoder = downsampling(in_channels, args.state_width)
    shape, vector_size = encoder.get_shape(data[0][0].unsqueeze(0))
    decoder = fc_layers(vector_size, nhidden, nclasses, True)
    drift = convolutions(args.state_width, args.width, shape)
    diffusion = None
    model = neural_de(drift, diffusion, encoder, decoder, backprop_option=args.adjoint_option)

    # optimiser and loss function
    optimiser = optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()
    val_metric = accuracy

    # run experiment
    run_experiment(args, args.dataset+'/'+str(args.batchsize), model, optimiser, train_loader, val_loader, loss_func, val_metric)



