import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from pyrates.utility.grid_search import grid_search
from copy import deepcopy
from scipy.ndimage.filters import gaussian_filter1d
from pyrates.utility.visualization import plot_timeseries, create_cmap
import h5py

linewidth = 1.2
fontsize1 = 10
fontsize2 = 10
markersize1 = 60
markersize2 = 60
dpi = 200

plt.style.reload_library()
plt.style.use('seaborn-whitegrid')
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']
#mpl.rc('text', usetex=True)
mpl.rcParams["font.sans-serif"] = ["Roboto"]
mpl.rcParams["font.size"] = fontsize1
mpl.rcParams["font.weight"] = "bold"
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rcParams['axes.titlesize'] = fontsize2
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['axes.labelsize'] = 'large'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['ytick.color'] = 'black'
mpl.rcParams['ytick.alignment'] = 'center'
mpl.rcParams['legend.fontsize'] = fontsize1
sns.set(style="whitegrid")

# parameter definitions
#######################

# simulation parameters
dt = 1e-3
dts = 1e-1
T = 2000.0

# model parameters
k_pp = 5.0
k_ap = 15.0
k_ep = 10.0
k_fp = 5.0

k_d1a = 5.0
k_d2a = 10.0

k_pe = 130.0
k_ae = 30.0

k_pd2 = 200.0
k_ad2 = 20.0
k_ad1 = 80.0
k_d1d1 = 5.0
k_d1d2 = 10.0
k_d2d2 = 8.0

k_d1f = 60.0
k_d2f = 40.0
k_ff = 10.0

param_grid = {
        'k_pe': [k_pe],
        'k_ae': [k_ae],
        'k_pp': [k_pp],
        'k_ap': [k_ap],
        'k_ep': [k_ep],
        'k_fp': [k_fp],
        'k_d1a': [k_d1a],
        'k_d2a': [k_d2a],
        'k_ad1': [k_ad1],
        'k_pd2': [k_pd2],
        'k_ad2': [k_ad2],
        'k_d1d1': [k_d1d1],
        'k_d1d2': [k_d1d2],
        'k_d2d2': [k_d2d2],
        'k_d1f': [k_d1f],
        'k_d2f': [k_d2f],
        'k_ff': [k_ff],
        'eta_p': [-3.0],
        'eta_a': [7.0],
        'eta_e': [0.5],
        'eta_d1': [-6.0],
        'eta_d2': [-6.0],
        'eta_fsi': [1.0],
        #'omega': stim_periods,
        #'alpha': np.asarray(stim_amps)
    }
param_grid = pd.DataFrame.from_dict(param_grid)

param_map = {
    'k_ae': {'vars': ['weight'], 'edges': [('stn', 'gpe_a')]},
    'k_pe': {'vars': ['weight'], 'edges': [('stn', 'gpe_p')]},
    'k_pp': {'vars': ['weight'], 'edges': [('gpe_p', 'gpe_p')]},
    'k_ap': {'vars': ['weight'], 'edges': [('gpe_p', 'gpe_a')]},
    'k_ep': {'vars': ['weight'], 'edges': [('gpe_p', 'stn')]},
    'k_fp': {'vars': ['weight'], 'edges': [('gpe_p', 'fsi')]},
    'k_d1a': {'vars': ['weight'], 'edges': [('gpe_a', 'msn_d1')]},
    'k_d2a': {'vars': ['weight'], 'edges': [('gpe_a', 'msn_d2')]},
    'k_pd2': {'vars': ['weight'], 'edges': [('msn_d2', 'gpe_p')]},
    'k_ad2': {'vars': ['weight'], 'edges': [('msn_d2', 'gpe_a')]},
    'k_d2d2': {'vars': ['weight'], 'edges': [('msn_d2', 'msn_d2')]},
    'k_ad1': {'vars': ['weight'], 'edges': [('msn_d1', 'gpe_a')]},
    'k_d1d1': {'vars': ['weight'], 'edges': [('msn_d1', 'msn_d1')]},
    'k_d1d2': {'vars': ['weight'], 'edges': [('msn_d2', 'msn_d1')]},
    'k_d1f': {'vars': ['weight'], 'edges': [('fsi', 'msn_d1')]},
    'k_d2f': {'vars': ['weight'], 'edges': [('fsi', 'msn_d2')]},
    'k_ff': {'vars': ['weight'], 'edges': [('fsi', 'fsi')]},
    'eta_p': {'vars': ['stn_op/eta'], 'nodes': ['gpe_p']},
    'eta_a': {'vars': ['stn_op/eta'], 'nodes': ['gpe_a']},
    'eta_e': {'vars': ['stn_op/eta'], 'nodes': ['stn']},
    'eta_d2': {'vars': ['msn_d1_op/eta'], 'nodes': ['msn_d2']},
    'eta_d1': {'vars': ['msn_d1_op/eta'], 'nodes': ['msn_d1']},
    'eta_fsi': {'vars': ['fsi_op/eta'], 'nodes': ['fsi']},
}

conditions = [{},  # healthy control -> GPe-p: 30 Hz, GPe-a: 3 Hz, STN: 6 Hz
              {'eta_d2': 30.0},  # STR excitation -> GPe-p: 3 Hz, GPe-a: 25 Hz, STN: 18 Hz
              {'eta_e': -20.0},  # STN inhibition -> GPe-p: 10 Hz, GPe_a: 12 Hz, STN: 1 Hz
              {'eta_p': 20.0},  # GPe-p excitation -> GPe-p: 100 Hz, GPe-a: 25 Hz
              #{'k_pp': 0.1, 'k_ap': 0.1, 'k_ps': 0.1, 'k_as': 0.1},  # GABAA blockade in GPe -> GPe_p: 70 Hz
              #{'k_pe': 0.1, 'k_pp': 0.1, 'k_ae': 0.1, 'k_ap': 0.1,
              # 'k_ps': 0.1, 'k_as': 0.1},  # AMPA blockade and GABAA blockade in GPe -> GPe_p: 35 Hz
              ]

# simulations
#############

outputs = {
            'stn': 'stn/stn_op/R',
            'gpe-p': 'gpe_p/stn_op/R',
            'gpe-a': 'gpe_a/stn_op/R',
            'msn-d1': 'msn_d1/msn_d1_op/R',
            'msn-d2': 'msn_d2/msn_d1_op/R',
            'fsi': 'fsi/fsi_op/R',
        }
for c_dict in deepcopy(conditions):
    for key in param_grid:
        if key in c_dict:
            if 'eta_' in key:
                c_dict[key] = np.asarray([c_dict[key]])
            else:
                c_dict[key] = np.asarray(param_grid[key]) * c_dict[key]
        elif key in param_grid:
            c_dict[key] = np.asarray(param_grid[key])
    param_grid_tmp = pd.DataFrame.from_dict(c_dict)
    results, result_map = grid_search(
        circuit_template="config/stn_gpe_str/stn_gpe_str_nodelay",
        param_grid=param_grid_tmp,
        param_map=param_map,
        simulation_time=T,
        step_size=dt,
        permute=True,
        sampling_step_size=dts,
        inputs={
            #'gpe_p/gpe_proto_syns_op/I_ext': ctx,
            #'gpe_a/gpe_arky_syns_op/I_ext': ctx
            },
        outputs=outputs.copy(),
        init_kwargs={
            'backend': 'numpy', 'solver': 'scipy', 'step_size': dt},
        method='RK45',
        clear=True
    )

    fig2, ax = plt.subplots(figsize=(6, 2.0), dpi=dpi)
    results = results * 1e3
    for key in outputs:
        ax.plot(results.loc[:, key])
    plt.legend(list(outputs.keys()))
    ax.set_ylabel('Firing rate')
    ax.set_xlabel('time (ms)')
    # ax.set_xlim([4000.0, 5000.0])
    # ax.set_ylim([0.0, 50.0])
    ax.tick_params(axis='both', which='major', labelsize=9)
    plt.tight_layout()
    plt.show()
