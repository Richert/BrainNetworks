from pyrates.utility import plot_timeseries, create_cmap, grid_search
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import correlate
import matplotlib as mpl

plt.style.reload_library()
plt.style.use('ggplot')
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.labelcolor'] = 'black'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['ytick.color'] = 'black'
mpl.rcParams['legend.fontsize'] = 12

# simulation parameters
dt = 1e-3
dts = 1e-1
T = 500.0

# model parameters
etas = np.linspace(-10.0, 10.0, num=50)
param_grid = {
        'k_ee': [43.4],
        'k_ae': [182.2],
        'k_pe': [550.6],
        'k_pp': [75.5],
        'k_ep': [588.4],
        'k_ap': [202.1],
        'k_aa': [252.8],
        'k_pa': [346.7],
        'k_ps': [797.4],
        'k_as': [805.0],
        'eta_e': np.asarray([1.25]),
        'eta_p': np.asarray([-6.65]) + etas,
        'eta_a': np.asarray([-13.74]),
        'delta_e': [0.646],
        'delta_p': [1.417],
        'delta_a': [1.068],
        'tau_e': [13],
        'tau_p': [25],
        'tau_a': [20],

    }

param_map = {
    'k_ee': {'vars': ['weight'], 'edges': [('stn', 'stn')]},
    'k_ae': {'vars': ['weight'], 'edges': [('stn', 'gpe_a')]},
    'k_pe': {'vars': ['weight'], 'edges': [('stn', 'gpe_p')]},
    'k_pp': {'vars': ['weight'], 'edges': [('gpe_p', 'gpe_p')]},
    'k_ep': {'vars': ['weight'], 'edges': [('gpe_p', 'stn')]},
    'k_ap': {'vars': ['weight'], 'edges': [('gpe_p', 'gpe_a')]},
    'k_aa': {'vars': ['weight'], 'edges': [('gpe_a', 'gpe_a')]},
    'k_pa': {'vars': ['weight'], 'edges': [('gpe_a', 'gpe_p')]},
    'k_ps': {'vars': ['weight'], 'edges': [('str', 'gpe_p')]},
    'k_as': {'vars': ['weight'], 'edges': [('str', 'gpe_a')]},
    'eta_e': {'vars': ['stn_op/eta_e'], 'nodes': ['stn']},
    'eta_p': {'vars': ['gpe_proto_op/eta_i'], 'nodes': ['gpe_p']},
    'eta_a': {'vars': ['gpe_arky_op/eta_a'], 'nodes': ['gpe_a']},
    'delta_e': {'vars': ['stn_op/delta_e'], 'nodes': ['stn']},
    'delta_p': {'vars': ['gpe_proto_op/delta_i'], 'nodes': ['gpe_p']},
    'delta_a': {'vars': ['gpe_arky_op/delta_a'], 'nodes': ['gpe_a']},
    'tau_e': {'vars': ['stn_op/tau_e'], 'nodes': ['stn']},
    'tau_p': {'vars': ['gpe_proto_op/tau_i'], 'nodes': ['gpe_p']},
    'tau_a': {'vars': ['gpe_arky_op/tau_a'], 'nodes': ['gpe_a']},
}

param_scalings = [
    ('delta_e', 'tau_e', 2.0),
    ('delta_p', 'tau_p', 2.0),
    ('delta_a', 'tau_a', 2.0),
    ('k_ee', 'delta_e', 0.5),
    ('k_ep', 'delta_e', 0.5),
    ('k_pe', 'delta_p', 0.5),
    ('k_pp', 'delta_p', 0.5),
    ('k_pa', 'delta_p', 0.5),
    ('k_ps', 'delta_p', 0.5),
    ('k_ae', 'delta_a', 0.5),
    ('k_ap', 'delta_a', 0.5),
    ('k_aa', 'delta_a', 0.5),
    ('k_as', 'delta_a', 0.5),
    ('eta_e', 'delta_e', 1.0),
    ('eta_p', 'delta_p', 1.0),
    ('eta_a', 'delta_a', 1.0),
]

# grid-search
#############

for key, key_tmp, power in param_scalings:
    param_grid[key] = np.asarray(param_grid[key]) * np.asarray(param_grid[key_tmp]) ** power

for key, val in param_grid.items():
    if len(val) == 1:
        param_grid[key] = np.asarray(list(val)*len(etas))

results, result_map = grid_search(
    circuit_template="config/stn_gpe/stn_gpe",
    param_grid=param_grid,
    param_map=param_map,
    simulation_time=T,
    step_size=dt,
    permute=False,
    sampling_step_size=dts,
    inputs={},
    outputs={
        'r_i': 'gpe_p/gpe_proto_op/R_i',
        #'r_e': 'stn/stn_op/R_e'
    },
    init_kwargs={'backend': 'numpy', 'solver': 'scipy', 'step_size': dt}
)
results = results * 1e3

# post-processing
#################
inputs, outputs = [], []
cutoff = 400.0

for i in range(len(etas)):

    idx = result_map.index[i]
    inputs.append(result_map.loc[idx, 'eta_p'])

    ts = results.loc[cutoff:, ('r_i', idx)].values
    outputs.append(np.mean(ts))

plt.plot(np.asarray(inputs), np.asarray(outputs))
plt.tight_layout()
plt.show()
