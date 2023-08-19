"""
Toy gradient test, testing the obtained gradients against analytical results for an exponential
Taken from ACA and MALI papers, adapted to include T as a parameter as well

have dz/dt = az, adn z(0) = z0, therefore z(T) = z0exp(aT)

Loss = z(T)^2 = (z0^2)exp(2aT), so gradients are

dL/da = 2T(z0^2)exp(2aT)
dL/dz0 = 2z0exp(2aT)
dL/dT = 2a(z0^2)exp(2aT)
"""


import argparse

import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--rtol', type=float, default=1e-6)
parser.add_argument('--atol', type=float, default=1e-7)
parser.add_argument('--gtol', type=float, default=0.4)
parser.add_argument('--method', type=str, choices=['gq', 'adjoint', 'direct', 'mali', 'aca'], default='gq')
parser.add_argument('--z0', type=float, default=10.0)
parser.add_argument('--a', type=float, default=0.2)
args = parser.parse_args()



# turn args into tensors
z0 = torch.tensor([args.z0]).float()
a = torch.tensor([args.a]).float()


# true loss functions and gradients
def loss_func(z):
    return z**2

def dldz(z, a, T):
    return 2*z*torch.exp(2*a*T)

def dlda(z, a, T):
    return 2*T*(z**2)*torch.exp(2*a*T)

def dldT(z, a, T):
    return 2*a*(z**2)*torch.exp(2*a*T)

def true_loss(z, a, T):
    return (z**2)*torch.exp(2*a*T)


# nn modules defining the odefunction and the odeblock
class odefunc(nn.Module):

    def __init__(self, a):
        super(odefunc, self).__init__()
        self.a = torch.nn.Parameter(a)
        self.nfe = 0

    def forward(self, t, z):
        self.nfe += 1
        return self.a*z


class odeblock(nn.Module):

    def __init__(self, a, z0, T, m):
        super(odeblock, self).__init__()
        self.func = odefunc(a)
        self.z0 = torch.nn.Parameter(z0)
        self.times = torch.nn.Parameter(torch.tensor([0.0, T]))
        self.method = m

    def forward(self):
        if self.method == 'mali' or self.method == 'aca':   # options for mali/aca from paper
            options = {}
            options['method'] = method
            options['h'] = None
            options['t0'] = self.times[0]
            options['t1'] = self.times[1]
            options['rtol'] = args.rtol
            options['atol'] = args.atol
            options['print_neval'] = False
            options['neval_max'] = 1000000
            if self.method == 'mali':
                options['regenerate_graph'] = False
                options['t_eval'] = None
                options['interpolation_method'] = 'cubic'
            elif self.method == 'aca':
                options['t_eval'] = self.times[1]
                options['interpolation_method'] = 'polynomial'
            out = odeint(self.func, self.z0, options = options)
        else:                                               # if using torchdiffeq methods
            options = {'rtol': args.rtol, 'atol': args.atol}
            if self.method == 'gq':
                options['gtol'] = args.gtol
            out = odeint(self.func, self.z0, self.times, **options)[1]
        return out


# get the errors for one particular T, print and add to a list
def get_error(T, t_list, loss_error_list, a_error_list, z0_error_list, T_error_list, m):
    model = odeblock(a, z0, T, m)
    zT = model()

    loss = loss_func(zT)
    loss.backward()

    agrad_pred = model.func.a.grad.item()
    z0grad_pred = model.z0.grad.item()
    Tgrad_pred = model.times.grad[1].item()

    loss_true = true_loss(z0, a, T).item()
    agrad_true = dlda(z0, a, T).item()
    z0grad_true = dldz(z0, a, T).item()
    Tgrad_true = dldT(z0, a, T).item()
    loss_error = loss_true - loss.item()
    a_error = agrad_true - agrad_pred
    z0_error = z0grad_true - z0grad_pred
    T_error = Tgrad_true - Tgrad_pred
    # we take the difference, can apply abs later/divide by true value later

    print('Method: {}, T: {:.2f}, loss error: {:.5f}, a error: {:.5f}, z0 error: {:.5f}, T error: {:.5f}'.format(m, T, loss_error, a_error, z0_error, T_error))
    t_list.append(T)
    loss_error_list.append(loss_error)
    a_error_list.append(a_error)
    z0_error_list.append(z0_error)
    T_error_list.append(T_error)


for m in ['gq', 'adjoint', 'direct', 'mali', 'aca']:

    # import correct torch ode solver
    if m == 'gq':
        from torch_gq_adjoint import odeint_adjoint_gq as odeint
    elif m =='adjoint':
        from torch_gq_adjoint import odeint_adjoint as odeint
    elif m == 'direct':
        from torch_gq_adjoint import odeint as odeint
    elif m == 'mali':
        from TorchDiffEqPack import odesolve_adjoint_sym12 as odeint
        method = 'sym12async'
    elif m =='aca':
        from TorchDiffEqPack import odesolve_adjoint as odeint
        method = 'Dopri5'


    # make linspace of Ts to test and set up save folder and empty lists of results
    Ts = np.arange(19.0, 29.0, 0.5)
    t_list = []
    loss_error_list = []
    a_error_list = []
    z0_error_list = []
    T_error_list = []
    folder = osp.join('results/', 'toy_gradient/', m)
    if not osp.exists(folder):
        os.makedirs(folder)

    # run experiment for each T
    for T in Ts:
        get_error(T, t_list, loss_error_list, a_error_list, z0_error_list, T_error_list, m)

    # save the results
    np.save(osp.join(folder, 'Ts.npy'), np.array(t_list))
    np.save(osp.join(folder, 'loss_errors.npy'), np.array(loss_error_list))
    np.save(osp.join(folder, 'a_errors.npy'), np.array(a_error_list))
    np.save(osp.join(folder, 'z0_errors.npy'), np.array(z0_error_list))
    np.save(osp.join(folder, 'T_errors.npy'), np.array(T_error_list))