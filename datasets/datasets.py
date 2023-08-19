"""
datasets used in the experiments
"""

import torch
import numpy as np
from math import pi
from torch.utils.data import Dataset
import torchvision.datasets as visiondatasets
import torchvision.transforms as transforms

import os
import os.path as osp


class g1d(Dataset):
    """
    Dataset of g1d task, simple regression task from Augmented Neural ODEs,
    https://arxiv.org/abs/1904.01681
    [-1, 1] ---> [1, -1].
    Parameters
    ----------
    t0 : float
        Initial time of the ODE
    t1 : float
        Final time of the ODE
    include_zero : Bool
        Can optionally add 0 to the problem, [-1, 0, 1] ---> [1, 0, -1].
    """
    def __init__(self, t0=0., t1=1., include_zero=False):
        self.num_samples = 2 + include_zero
        self.data = []
        times = torch.tensor([[t0], [t1]]).float()
        xs = torch.tensor([[-1.], [1.]]).float()
        if include_zero:
            xs = torch.cat((xs, torch.tensor([[0.]]).float()), dim=0)

        for i in range(self.num_samples):
            x = xs[i]
            y = -x
            self.data.append((x, times, y))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples


class nested_spheres(Dataset):
    """
    2D Dataset of nested spheres task, simple classification task from Augmented Neural ODEs,
    if x < r1: y = 0,       if r1 < x < r2: y =1
    Parameters
    ----------
    t0 : float
        Initial time of the ODE
    t1 : float
        Final time of the ODE
    train : Bool
        Create a train set or a validaton set.
    """
    def __init__(self, t0=0., t1=1., train=True):
        folder = 'datasets/data/nested_spheres/'
        
        if train:
            x_data = torch.load(osp.join(folder, 'nested_spheres_x_train.pt'))
            y_data = torch.load(osp.join(folder, 'nested_spheres_y_train.pt'))
        else:
            x_data = torch.load(osp.join(folder, 'nested_spheres_x_val.pt'))
            y_data = torch.load(osp.join(folder, 'nested_spheres_y_val.pt'))
        
        self.num_samples = len(x_data)
        times = torch.tensor([[t0], [t1]]).float()
        self.data = []
        
        for i in range(self.num_samples):
            x = x_data[i]
            y = y_data[i]
            self.data.append((x, times, y))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples


class sinedata(Dataset):
    """
    Toy dataset for the sines times series problem, where x'' = -w^2x
    Parameters
    ----------
    train: Bool
        Set to true for training set, false for validation set
    regular: Bool
        Set to true to have regular times (same regular times across batches),
        set to false for irregular times (different across batches)
    """
    def __init__(self, train=True, ntimes=10, regular=True):
        folder = 'datasets/data/sines/'

        if train:
            x0_data = torch.load(osp.join(folder, 'sines_xv0_'+str(ntimes)+'_train.pt'))
            t_data = torch.load(osp.join(folder, 'sines_t_'+str(ntimes)+'_train.pt'))
        else:
            x0_data = torch.load(osp.join(folder, 'sines_xv0_'+str(ntimes)+'_val.pt'))
            t_data = torch.load(osp.join(folder, 'sines_t_'+str(ntimes)+'_val.pt'))

        self.num_samples = len(x0_data)
        self.data = []

        for i in range(self.num_samples):
            x = x0_data[i]
            if regular:
                t = torch.linspace(0, 2*pi, ntimes).unsqueeze(-1)
            else:
                t = t_data[i]
            B = x[0]
            A = x[1]
            y = A*torch.sin(t) + B*torch.cos(t)
            self.data.append((x, t, y))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples


class image_data(Dataset):
    """
    Dataset to get images for classification, includes MNIST, CIFAR10, SVHN
    parameters
    ----------
    t0: float
        starting time
    t1: float
        end time
    dataset: str
        which dataset to use
    train: bool
        whether to use the train set or not
    """
    def __init__(self, t0=0., t1=1., dataset='mnist', train=True):
        folder = osp.join('datasets/data/', dataset)
        
        if dataset == 'mnist':
            data = visiondatasets.MNIST(root=folder, train=train, transform=transforms.transforms.ToTensor(), download=True)
        elif dataset == 'cifar10':
            data = visiondatasets.CIFAR10(root=folder, train=train, transform=transforms.transforms.ToTensor(), download=True)
        elif dataset =='svhn':
            split = 'train' if train else 'test'
            data = visiondatasets.SVHN(root=folder, split=split, transform=transforms.transforms.ToTensor(), download=True)

        self.num_samples = len(data)
        times = torch.tensor([[t0], [t1]]).float()
        self.data = []
        
        for i in range(self.num_samples):
            self.data.append((data[i][0], times, data[i][1]))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples


class oudata(Dataset):
    """
    Data from the Ornstein Uhlenbeck Process
    Parameters
    ----------
    train: bool
        Whether it is the training or val data
    """
    def __init__(self, train=True):
        folder = 'datasets/data/ornstein_uhlenbeck/'

        if train:
            y_data = torch.load(osp.join(folder, 'ou_y_train.pt'))
            t_data = torch.load(osp.join(folder, 'ou_t_train.pt'))     
        else:
            y_data = torch.load(osp.join(folder, 'ou_y_val.pt'))
            t_data = torch.load(osp.join(folder, 'ou_t_val.pt'))

        self.num_samples = len(y_data)
        self.data = []

        for i in range(self.num_samples):
            y = y_data[i]
            x = y[:, 0, :]
            self.data.append((x, t_data.repeat(len(x), 1, 1), y))  #have to repeat t because each x needs times

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples