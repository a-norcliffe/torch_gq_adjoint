"""
Neural ODE and SDE Helper Classes, for the experiments specifically, might
not work well in a general setting, we encourage to use the
torch_gq_adjoint library for that.
"""

import torch
import torch.nn as nn
import numpy as np
from torch_gq_adjoint import odeint, odeint_adjoint, odeint_adjoint_gq, make_seminorm
from torchsde import sdeint, sdeint_adjoint, BrownianInterval


class make_ode_func(nn.Module):
    """
    Makes an ODE function for the neural differential equation
    Parameters
    ----------
    func: nn.Module
        The ODE function
    """
    def __init__(self, func):
        super(make_ode_func, self).__init__()
        self.func = func
        self.nfe = 0
    
    def forward(self, t, x):
        self.nfe += 1
        return self.func(t, x)


class make_sde_func(nn.Module):
    """
    Makes an SDE function for the neural differential equation,
    needs to be slightly more complicated than the ode maker to
    incorporate the Wong Zakai Training
    Parameters
    ----------
    drift: nn.Module
        The SDE drift
    diffusion: nn.Module
        The SDE diffusion
    """
    def __init__(self, drift, diffusion):
        super(make_sde_func, self).__init__()
        self.f = drift
        self.g = diffusion
        self.noise_type = 'diagonal'
        self.sde_type = 'stratonovich'
        self.ncosines = 25
        self.seed = 10000*np.random.randn()
        self.t0 = 0  # Can be replaced by the specific dataset
        self.t1 = 1  # Can be replaced by the specific dataset
        self.pi = torch.tensor(np.pi)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        return self.f(t, x) + self.g(t, x)*self.noise(t, x)

    def noise(self, t, x):
        torch.manual_seed(self.seed)
        c = self.t1 - self.t0
        t = (t-self.t0)/c
        x = torch.zeros_like(x)
        for n in range(1, self.ncosines+1):
            x += torch.randn_like(x)*(torch.cos((n-0.5)*self.pi*t).item())
        return x*(2/c)**0.5


class neural_de(nn.Module):
    """
    Neural DE class, can do drift and diffusion, set diffusion to None for ODE
    Parameters
    ----------
    drift: nn.Module
        A torch module which acts as the ODE function, or the SDE drift
    diffusion: nn.Module
        A torch module which acts as the SDE diffusion
    encoder: nn.Module
        A torch module which encodes the initial condition into latent space
    decoder: nn.Module
        A torch module which decodes the ODE solution to observation space
    backprop_option: str
        Decides which method for calculating gradients to use (direct, adjoint, gq, seminorm)
        Is also used to use torchsde for sde experiments
    """
    def __init__(self, drift, diffusion, encoder, decoder, backprop_option='adjoint_gq'):
        super(neural_de, self).__init__()
        
        if diffusion is None:
            self.defunc = make_ode_func(drift)
            self.detype = 'ode'
        else:
            self.defunc = make_sde_func(drift, diffusion)
            self.detype = 'kl_sde'
            self.rev_heun_stepsize = 0.01   #NOTE can change the step size here, 0.01 is specfic to the experiments run

        self.encoder = encoder
        self.decoder = decoder
        self.use_seminorm = False

        if backprop_option == 'direct':
            self.odeint = odeint
        elif backprop_option == 'adjoint_ode':
            self.odeint = odeint_adjoint
        elif backprop_option == 'adjoint_seminorm':
            self.odeint = odeint_adjoint
            self.use_seminorm = True
        elif backprop_option == 'adjoint_gq':
            self.odeint = odeint_adjoint_gq
        elif backprop_option == 'sde_direct':
            self.odeint = sdeint
            self.detype = 'true_sde'
        elif backprop_option == 'sde_adjoint':
            self.odeint = sdeint_adjoint
            self.detype = 'true_sde'
            

    def forward(self, x, times, **model_kwargs):
        x = self.encoder(x)

        # required for seminorms if they are used
        if self.use_seminorm:
            model_kwargs['adjoint_options'] = dict(norm=make_seminorm(x))

        # used to sort the batches of times, only works for vector 
        # (ODEfunc must unflatten and reflatten for convolutions)
        times = times.reshape(-1, times.size(-2), times.size(-1))
        times, ind = times.unique(sorted=True, return_inverse=True) 
        ind = ind.expand(ind.size(0), ind.size(1), x.size(-1))

        # if sde need to scale the brownian motion
        if self.detype == 'kl_sde':
            self.defunc.t0 = times[0]
            self.defunc.t1 = times[-1]
            self.defunc.seed += 500

        # choose reversible heun if true sde
        if self.detype == 'true_sde':
            model_kwargs['method'] = 'reversible_heun'
            model_kwargs['adjoint_method'] = 'adjoint_reversible_heun'
            model_kwargs['dt'] = self.rev_heun_stepsize
        else:
            model_kwargs['method'] = 'dopri5'
        
        # solve the ODE/SDE
        x = self.odeint(self.defunc, x, times, **model_kwargs)

        # reshape so batch is first dimension and gather the correct times in the batch
        x = x.transpose(0, 1)                              
        x = x.gather(dim=1, index=ind)
        x = self.decoder(x)
        return x


    def evaluate(self, x, times, **kwargs):
        if (self.detype == 'ode') or (self.detype == 'true_sde'):
            with torch.no_grad():
                return self.forward(x, times, **kwargs)

        elif self.detype == 'kl_sde':   
            #uses same weights but transfers to an SDE solver, rather than Karhunen Loeve SDE
            with torch.no_grad():
                x = self.encoder(x)

                # sort the batches (as above)
                times = times.reshape(-1, times.size(-2), times.size(-1))
                times, ind = times.unique(sorted=True, return_inverse=True) 
                ind = ind.expand(ind.size(0), ind.size(1), x.size(-1))

                # delete gtol if it is present to prevent error message
                try:
                    del kwargs['gtol']
                except KeyError:
                    pass

                # solve the SDE with given stepsize
                kwargs['dt'] = self.rev_heun_stepsize
                x = sdeint(self.defunc, x, times, method='reversible_heun', **kwargs)

                # reshape as above
                x = x.transpose(0, 1)                                       
                x = x.gather(dim=1, index=ind)
                                            
                x = self.decoder(x)
                return x


