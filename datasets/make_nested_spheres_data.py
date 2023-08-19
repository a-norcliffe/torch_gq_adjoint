"""
Nested Spheres data
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import os
import os.path as osp

if not osp.exists('datasets/data/nested_spheres'):
    os.makedirs('datasets/data/nested_spheres')

for name in ['train', 'val']:

    if name=='train':
        npoints = 1000
    else:
        npoints = 50

    r1 = 0.4
    r2 = 0.7
    r3 = 0.9

    x_data0 = []
    y_data0 = []
    x_data1 = []
    y_data1 = []

    while (len(x_data0) < npoints) or (len(x_data1) < npoints):
        x = np.random.uniform(low=-1, high=1, size=2)
        length = np.linalg.norm(x)
        if length < r1:
            x_data0.append(x)
            y_data0.append(0)
        elif r2 < length < r3 :
            x_data1.append(x)
            y_data1.append(1)
        else:
            pass

    x_data0 = np.asarray(x_data0[:npoints])
    y_data0 = np.asarray(y_data0[:npoints])    
    x_data1 = np.asarray(x_data1[:npoints])
    y_data1 = np.asarray(y_data1[:npoints])   

    x_data = np.concatenate((x_data0, x_data1), axis=0)
    y_data = np.concatenate((y_data0, y_data1), axis=0)

    x_data = torch.tensor(x_data).float()
    y_data = torch.tensor(y_data).long()

    torch.save(x_data, 'datasets/data/nested_spheres/nested_spheres_x_'+name+'.pt')
    torch.save(y_data, 'datasets/data/nested_spheres/nested_spheres_y_'+name+'.pt')




