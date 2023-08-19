"""
make sine initial conditions and times
"""

import torch
import argparse
from numpy import pi
import os
import os.path as osp


if not osp.exists('datasets/data/sines'):
    os.makedirs('datasets/data/sines')


for name in ['train', 'val']:
    for ntimes in [10, 50]:

      if name=='train':
          nsines = 15
      else:
          nsines = 5
    
    
      xv = torch.empty((nsines, 2))
      t = torch.empty((nsines, ntimes, 1))


      for i in range(nsines):
          initial_cond = 2*torch.rand(2)-1
          xv[i] = initial_cond
          times, _ = (2*pi*torch.rand(ntimes)).sort()
          times[0] = 0
          times = times.unsqueeze(-1)
          t[i] = times
      
      xv = xv.float()
      t = t.float()

      torch.save(xv, 'datasets/data/sines/sines_xv0_'+str(ntimes)+'_'+name+'.pt')
      torch.save(t, 'datasets/data/sines/sines_t_'+str(ntimes)+'_'+name+'.pt')

