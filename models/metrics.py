"""
Metrics such as accuracy for classification, KL divergence etc. can be used for Training
or Validation
"""

import torch
import numpy as np
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal


def accuracy(y_pred, y):
    y_pred = torch.argmax(y_pred, dim=-1)
    acc = 100*torch.sum(y_pred==y)/len(y)
    return acc


def convert_to_normal(x, sigma=1e-9, keep_shape=False):
    """Converts set of predictions to a normal distirbution by taking mean and 
    std"""
    mean = torch.mean(x, dim=1)
    std = torch.std(x, dim=1) + sigma #add this to avoid delta distributions
    
    if keep_shape:      # if we want to calculate log probability for the SDE
        s = tuple(x.shape)
        mean_temp = torch.empty(s).to(x.device)
        std_temp = torch.empty(s).to(x.device)
        for i in range(s[0]):
            for j in range(s[1]):
                mean_temp[i][j] = mean[i]
                std_temp[i][j] = std[i]  
        mean = mean_temp
        std = std_temp
    
    dist = Normal(mean, std)
    return dist


def kl_divergence_samples(y_pred, y):
    y_pred = y_pred.reshape(y.shape)
    y_pred_dist = convert_to_normal(y_pred)
    y_dist = convert_to_normal(y)
    kl = kl_divergence(y_pred_dist, y_dist).mean()
    return kl


def sample_logp(y_pred, y):
    # calculates the mean, can multiply by number of points if required
    y_pred = y_pred.reshape(y.shape)
    y_pred_dist = convert_to_normal(y_pred, keep_shape=True)
    logp = y_pred_dist.log_prob(y).mean()
    return logp


def kl_divergence_samples_training_range(y_pred, y):
    start = 0
    end = 300
    y_pred = y_pred.reshape(y.shape)
    y_pred = y_pred[:, :, start:end, :]
    y = y[:, :, start:end, :]
    y_pred_dist = convert_to_normal(y_pred, sigma=2e-3)
    y_dist = convert_to_normal(y, sigma=2e-3)
    kl = kl_divergence(y_pred_dist, y_dist).mean()
    return kl


def kl_divergence_samples_testing_range(y_pred, y):
    start = 300
    end = 450
    y_pred = y_pred.reshape(y.shape)
    y_pred = y_pred[:, :, start:end, :]
    y = y[:, :, start:end, :]
    y_pred_dist = convert_to_normal(y_pred, sigma=2e-3)
    y_dist = convert_to_normal(y, sigma=2e-3)
    kl = kl_divergence(y_pred_dist, y_dist).mean()
    return kl