import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from pyrates.utility.grid_search import grid_search
from copy import deepcopy

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
k_11 = 5.0
k_12 = 10.0
k_22 = 8.0

k_1f = 60.0
k_2f = 40.0
k_ff = 10.0

param_grid = {
        'k_11': [k_11],
        'k_12': [k_12],
        'k_22': [k_22],
        'k_1f': [k_1f],
        'k_2f': [k_2f],
        'k_ff': [k_ff],
        'eta_1': [-7.0],
        'eta_2': [-7.0],
        'eta_f': [0.4],
    }
param_grid = pd.DataFrame.from_dict(param_grid)

param_map = {
    'k_11': {'vars': ['weight'], 'edges': [('msn_d1', 'msn_d1')]},
    'k_12': {'vars': ['weight'], 'edges': [('msn_d2', 'msn_d1')]},
    'k_22': {'vars': ['weight'], 'edges': [('msn_d2', 'msn_d2')]},
    'k_1f': {'vars': ['weight'], 'edges': [('fsi', 'msn_d1')]},
    'k_2f': {'vars': ['weight'], 'edges': [('fsi', 'msn_d2')]},
    'k_ff': {'vars': ['weight'], 'edges': [('fsi', 'fsi')]},
    'eta_1': {'vars': ['msn_d1_op/eta'], 'nodes': ['msn_d1']},
    'eta_2': {'vars': ['msn_d1_op/eta'], 'nodes': ['msn_d2']},
    'eta_f': {'vars': ['fsi_op/eta'], 'nodes': ['fsi']},
}

conditions = [{},  # healthy control -> MSN-D1: 1 Hz, MSN-D2: 2 Hz, FSI: 10 Hz
              {'eta_f': -30.0},  # FSI inhibition -> MSN-D1: 2 Hz, MSN-D2: 4 Hz, FSI: 0 Hz
              ]

# simulations
#############

outputs = {
            'msn-d1': 'msn_d1/msn_d1_op/R',
            'msn-d2': 'msn_d2/msn_d1_op/R',
            'fsi': 'fsi/fsi_op/R'
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
        circuit_template="config/stn_gpe_str/str",
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
    ax.set_xlim([1000.0, 2000.0])
    ax.set_ylim([0.0, 20.0])
    ax.tick_params(axis='both', which='major', labelsize=9)
    plt.tight_layout()
    plt.show()
