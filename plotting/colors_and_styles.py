"""
common colours and linestyles etc. for plotting
"""

import matplotlib.pyplot as plt
import seaborn as sns

colors = sns.color_palette("tab10")


method_colors = {'direct': colors[2],
                 'adjoint': colors[6],
                 'seminorm': colors[1],
                 'gq': colors[0],
                 'mali': colors[1],
                 'aca': colors[3],
                 'sde_adjoint': colors[6],
                 'sde_direct': colors[2]}


method_linestyles = {'direct': '-',
                     'adjoint': '--',
                     'seminorm': '-.',
                     'gq': ':',
                     'mali': '-.',
                     'aca': (0, (3, 1, 1, 1)),
                     'sde_adjoint': '--',
                     'sde_direct': '-'}


method_markers = {'direct': 'D',
                 'adjoint': '^',
                 'seminorm': 'x',
                 'gq': 'o',
                 'mali': 'x',
                 'aca': '+',
                 'sde_adjoint': '^',
                 'sde_direct': 'D'}


method_names = {'direct': 'Direct',
                 'adjoint': 'Adjoint',
                 'seminorm': 'Seminorm',
                 'gq': 'GQ',
                 'mali': 'MALI',
                 'aca': 'ACA',
                 'sde_adjoint': 'SDE Adjoint',
                 'sde_direct': 'SDE Direct'}


method_filenames = {'direct': 'direct',
                    'adjoint': 'adjoint_ode',
                    'seminorm': 'adjoint_seminorm',
                    'gq': 'adjoint_gq',
                    'sde_adjoint': 'sde_adjoint',
                    'sde_direct': 'sde_direct'}


colors_dict =   {'green': colors[2],
                 'pink': colors[6],
                 'orange': colors[1],
                 'blue': colors[0],
                 'red': colors[3]}


linestyles_dict   = {'standard': '-',
                     'dashed': '--',
                     'dashdotted': '-.',
                     'dotted': ':',
                     'densedashdotted': (0, (3, 1, 1, 1))}