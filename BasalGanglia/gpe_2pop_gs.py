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
T = 5000.0
sim_steps = int(np.round(T/dt))
stim_offset = int(np.round(600.0/dt))
stim_dur = int(np.round(50.0/dt))
stim_delayed = int(np.round((1200.0)/dt))
stim_amp = 10.0
stim_var = 10.0
stim_period = 77
stim_periods = [stim_period]
stim_amps = [15.0]

ctx = np.zeros((sim_steps, 1))
ctx[stim_offset:stim_offset+stim_dur, 0] = 5.0*100.0
ctx[stim_delayed:stim_delayed+stim_dur, 0] = -5.0*100.0
ctx = gaussian_filter1d(ctx, stim_var, axis=0)

# plt.plot(ctx)
# plt.show()

# model parameters
k_gp = 1.0
k = 10.0
param_grid = {
        'k_ae': [k*1.5],
        'k_pe': [k*5.0],
        'k_pp': [5.0*k*k_gp],
        'k_ap': [2.0*k*k_gp],
        'k_aa': [0.1*k*k_gp],
        'k_pa': [0.5*k*k_gp],
        'k_ps': [10.0*k*k_gp],
        'k_as': [1.0*k*k_gp],
        'eta_e': [0.02],
        'eta_p': [40.0],
        'eta_a': [26.0],
        'eta_s': [0.002],
        'delta_p': [9.0],
        'delta_a': [3.0],
        'tau_p': [18],
        'tau_a': [32],
        #'omega': stim_periods,
        #'alpha': np.asarray(stim_amps)
    }
param_grid = pd.DataFrame.from_dict(param_grid)

param_map = {
    'k_ae': {'vars': ['weight'], 'edges': [('stn', 'gpe_a')]},
    'k_pe': {'vars': ['weight'], 'edges': [('stn', 'gpe_p')]},
    'k_pp': {'vars': ['weight'], 'edges': [('gpe_p', 'gpe_p')]},
    'k_ap': {'vars': ['weight'], 'edges': [('gpe_p', 'gpe_a')]},
    'k_aa': {'vars': ['weight'], 'edges': [('gpe_a', 'gpe_a')]},
    'k_pa': {'vars': ['weight'], 'edges': [('gpe_a', 'gpe_p')]},
    'k_ps': {'vars': ['weight'], 'edges': [('str', 'gpe_p')]},
    'k_as': {'vars': ['weight'], 'edges': [('str', 'gpe_a')]},
    'eta_p': {'vars': ['gpe_proto_syns_op/eta_i'], 'nodes': ['gpe_p']},
    'eta_a': {'vars': ['gpe_arky_syns_op/eta_a'], 'nodes': ['gpe_a']},
    'eta_e': {'vars': ['stn_dummy_op/eta_e'], 'nodes': ['stn']},
    'eta_s': {'vars': ['str_dummy_op/eta_s'], 'nodes': ['str']},
    'delta_p': {'vars': ['gpe_proto_syns_op/delta_i'], 'nodes': ['gpe_p']},
    'delta_a': {'vars': ['gpe_arky_syns_op/delta_a'], 'nodes': ['gpe_a']},
    'tau_p': {'vars': ['gpe_proto_syns_op/tau_i'], 'nodes': ['gpe_p']},
    'tau_a': {'vars': ['gpe_arky_syns_op/tau_a'], 'nodes': ['gpe_a']},
    #'omega': {'vars': ['sl_op/t_off'], 'nodes': ['driver']},
    #'alpha': {'vars': ['sl_op/alpha'], 'nodes': ['driver']}
}

conditions = [{},  # healthy control -> GPe-p: 60 Hz, GPe-a: 10 Hz
              {'eta_s': 20.0},  # STR excitation -> GPe-p: 10 Hz, GPe-a: 40 Hz
              {'eta_e': 0.1},  # STN inhibition -> GPe-p: 30 Hz, GPe_a: 20 Hz
              {'k_pp': 0.1, 'k_pa': 0.1, 'k_aa': 0.1, 'k_ap': 0.1, 'k_ps': 0.1,
               'k_as': 0.1},  # GABAA blockade in GPe -> GPe_p: 100 Hz
              {'k_pe': 0.1, 'k_pp': 0.1, 'k_pa': 0.1, 'k_ae': 0.1, 'k_aa': 0.1, 'k_ap': 0.1,
               'k_ps': 0.1, 'k_as': 0.1},  # AMPA blockade and GABAA blockade in GPe -> GPe_p: 70 Hz
              ]

# simulations
#############

for c_dict in deepcopy(conditions):
    for key in param_grid:
        if key in c_dict:
            c_dict[key] = np.asarray(param_grid[key]) * c_dict[key]
        elif key in param_grid:
            c_dict[key] = np.asarray(param_grid[key])
    param_grid_tmp = pd.DataFrame.from_dict(c_dict)
    results, result_map = grid_search(
        circuit_template="config/stn_gpe/gpe_2pop",
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
        outputs={
            'r_i': 'gpe_p/gpe_proto_syns_op/R_i',
            'r_a': 'gpe_a/gpe_arky_syns_op/R_a',
        },
        init_kwargs={
            'backend': 'numpy', 'solver': 'scipy', 'step_size': dt},
        method='RK45',
    )

    fig2, ax = plt.subplots(figsize=(6, 2.0), dpi=dpi)
    results = results * 1e3
    plot_timeseries(results, ax=ax)
    plt.legend(['GPe-p', 'GPe-a'])
    ax.set_ylabel('Firing rate')
    ax.set_xlabel('time (ms)')
    # ax.set_xlim([000.0, 1500.0])
    # ax.set_ylim([0.0, 100.0])
    ax.tick_params(axis='both', which='major', labelsize=9)
    plt.tight_layout()
    plt.show()
