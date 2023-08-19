"""
Common nn Modules that are shared across experiments
"""

import torch
import torch.nn as nn



class mlp(nn.Module):
    """
    Multilayer Perceptron with x only as input, 2 hidden layers
    Parameters
    ----------
    dim: int
        The dimension of input data, i.e. dim(x)
    nhidden: int
        The width of the hidden layer
    """
    def __init__(self, dim, nhidden):
        super(mlp, self).__init__()
        self.act = nn.Softplus()
        self.fc1 = nn.Linear(dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, dim)

    def forward(self, t, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x


class mlp_t(nn.Module):
    """
    Multilayer Perceptron with x and t as input, 2 hidden layers
    Parameters
    ----------
    dim: int
        The dimension of input data, i.e. dim(x)
    nhidden: int
        The width of the hidden layer
    """
    def __init__(self, dim, nhidden):
        super(mlp_t, self).__init__()
        self.act = nn.Softplus()
        self.fc1 = nn.Linear(1+dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, dim)

    def forward(self, t, x):
        size = list(x.size())
        size[-1] = 1
        x = torch.cat((x, t.repeat(size)), dim=-1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x


class mlp_sonode(nn.Module):
    """
    Multilayer perceptron, where the dynamics are second order, acts on [x, v]
    2 hidden layers
    Parameters
    ----------
    dim: int
        Dimension of x: dim(x)
    nhidden: int
        Width of the hidden dimension
    """
    def __init__(self, dim, nhidden):
        super(mlp_sonode, self).__init__()
        self.xdim = dim
        self.act = nn.Softplus()
        self.fc1 = nn.Linear(2*dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, dim)

    def forward(self, t, z):
        v = z[..., self.xdim:]
        z = self.fc1(z)
        z = self.act(z)
        z = self.fc2(z)
        z = self.act(z)
        z = self.fc3(z)
        return torch.cat((v, z), dim=-1)


class identity(nn.Module):
    """
    Module that simply does nothing, useful in the general framework, where we
    need an encoder, even if it is just an identity mapping
    """
    def __init__(self):
        super(identity, self).__init__()

    def forward(self, x):
        return x


class zero_aug(nn.Module):
    """
    Encoder module which augments the state (a vector) with zeros
    Parameters
    ----------
    aug_dim
        Number of dimensions to augment x with
    """
    def __init__(self, aug_dim):
        super(zero_aug, self).__init__()
        self.aug_dim = aug_dim

    def forward(self, x):
        zeros = torch.zeros((len(x), self.aug_dim)).to(x.device)
        return torch.cat((x, zeros), dim=-1)


class remove_aug(nn.Module):
    """
    Decoder for augmented odes, will pick out only
    the dimensions that were not augmented
    Parameters
    ----------
    x_dim: int
        The dimension of input data, i.e. dim(x)
    last_only: Bool
        Whether we take the final value only of the ODE or not
    """
    def __init__(self, x_dim, last_only=False):
        super(remove_aug, self).__init__()
        self.x_dim = x_dim
        self.last_only = last_only

    def forward(self, x):
        if self.last_only:
            return x[:, -1, :self.x_dim].reshape(-1, self.x_dim)
        else:
            return x[:, :, :self.x_dim]


class linear_layer(nn.Module):
    """
    Linear layer, can be used for encoding or decoding as in 
    Dissecting Neural ODEs
    Parameters
    ----------
    in_dim: int
        The dimension of input data, i.e. dim(x)
    out_dim: int
        The dimension of output, e.g. number of classes
    last_only: Bool
        Whether we take the final value only of the ODE or not
    """
    def __init__(self, in_dim, out_dim, last_only=False):
        super(linear_layer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.last_only = last_only

    def forward(self, x):
        if self.last_only:
            x = x[:, -1, :]
        return self.linear(x)

    
class flattener(nn.Module):
    """
    Flattens images to vectors
    Parameters
    ----------
    shape: tensor of ints, of one batch of x
    """
    def __init__(self, shape):
        super(flattener, self).__init__()
        self.shape = torch.prod(shape).item()

    def forward(self, x):
        return x.view(-1, self.shape)


class unflattener(nn.Module):
    """
    Unflattens vectors back to images
    Parameters
    ----------
    shape: tensor of ints, of one batch of x
    """
    def __init__(self, shape):
        super(unflattener, self).__init__()
        self.shape = shape.tolist()
        self.shape = [-1] + self.shape

    def forward(self, x):
        return x.view(self.shape)


class convolutions(nn.Module):
    """
    ODEfunc that runs convolutions on images
    Parameters
    ----------
    in_channels: int
        number of channels of input image
    nhidden: int
        the number of hidden channels
    shape: tensor of ints
        shape of a single image
    """
    def __init__(self, in_channels, nhidden, shape):
        super(convolutions, self).__init__()
        self.activation = nn.Softplus()
        self.conv1 = nn.Conv2d(in_channels, nhidden, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(nhidden, nhidden, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(nhidden, in_channels, kernel_size=3, stride=1, padding=1)
        self.unflatten = unflattener(shape)
        self.flatten = flattener(shape)

    def forward(self, t, x):
        x = self.unflatten(x)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.flatten(x)
        return x


class downsampling(nn.Module):
    """
    downsampling method before ODE solve for image classification
    Parameters
    ----------
    in_channels: number of channels of original image
    """
    def __init__(self, in_channels, nhidden):
        super(downsampling, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm2d(in_channels)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(in_channels, nhidden, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(nhidden, nhidden, kernel_size=3, stride=1, padding=0)
        self.flatten = identity() #have to set to identity first, before getting the shape

    def forward(self, x):
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.flatten(x)
        return x

    def get_shape(self, x):
        x = self.forward(x)
        x = torch.tensor(x.size()[1:]).int()
        self.flatten = flattener(x)
        return x, torch.prod(x).item()


class fc_layers(nn.Module):
    """
    1 hidden layer mlp for encoding or decoding, set last_only = True to only take final state
    Parameters
    ----------
    in_dim: int
        dimension of vector going into MLP
    nhidden: int
        dimension of hidden layer
    out_dim: int
        dimension of final layer
    """
    def __init__(self, in_dim, nhidden, out_dim, last_only=False):
        super(fc_layers, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(in_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, out_dim)
        self.last_only = last_only

    def forward(self, x):
        if self.last_only:
            x = x[:, -1, :]
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class squeezedim1(nn.Module):
    """
    Squeezes dim1 add encoder if necessary
    """
    def __init__(self, encoder):
        super(squeezedim1, self).__init__()
        if encoder is None:
            self.use_encoder = False
        else:
            self.use_encoder = True
            self.encoder = encoder
            
    def forward(self, x):
        vectorsize = x.size(-1)
        x = x.reshape(-1, vectorsize)
        if self.use_encoder:
            x = self.encoder(x)
        return x
