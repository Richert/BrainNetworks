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
T = 200.0

# model parameters
etas = np.linspace(-10.0, 40.0, num=50)
ks = np.asarray([0.5, 0.75, 1.0, 1.25, 1.5])
k_0 = 10.0
param_grid = {
        'k_ee': np.asarray([k_0]) * ks,
        'k_ae': np.asarray([100.0]),
        'k_pe': np.asarray([100.0]),
        'k_pp': np.asarray([20.0]),
        'k_ep': np.asarray([80.0]),
        'k_ap': np.asarray([80.0]),
        'k_aa': np.asarray([5.0]),
        'k_pa': np.asarray([20.0]),
        'k_ps': np.asarray([100.0]),
        'k_as': np.asarray([200.0]),
        'eta_e': np.asarray([0.0]) + etas,
        'eta_p': np.asarray([0.0]),
        'eta_a': np.asarray([2.0]),
        'delta_e': np.asarray([0.1]),
        'delta_p': np.asarray([0.1]),
        'delta_a': np.asarray([0.2]),
        'tau_e': np.asarray([13]),
        'tau_p': np.asarray([25]),
        'tau_a': np.asarray([20]),

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
    'eta_e': {'vars': ['stn_syns_op/eta_e'], 'nodes': ['stn']},
    'eta_p': {'vars': ['gpe_proto_syns_op/eta_i'], 'nodes': ['gpe_p']},
    'eta_a': {'vars': ['gpe_arky_syns_op/eta_a'], 'nodes': ['gpe_a']},
    'delta_e': {'vars': ['stn_syns_op/delta_e'], 'nodes': ['stn']},
    'delta_p': {'vars': ['gpe_proto_syns_op/delta_i'], 'nodes': ['gpe_p']},
    'delta_a': {'vars': ['gpe_arky_syns_op/delta_a'], 'nodes': ['gpe_a']},
    'tau_e': {'vars': ['stn_syns_op/tau_e'], 'nodes': ['stn']},
    'tau_p': {'vars': ['gpe_proto_syns_op/tau_i'], 'nodes': ['gpe_p']},
    'tau_a': {'vars': ['gpe_arky_syns_op/tau_a'], 'nodes': ['gpe_a']},
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

results, result_map = grid_search(
    circuit_template="config/stn_gpe/stn_gpe_syns",
    param_grid=param_grid,
    param_map=param_map,
    simulation_time=T,
    step_size=dt,
    permute_grid=True,
    sampling_step_size=dts,
    inputs={},
    outputs={
        'r_i': 'gpe_p/gpe_proto_syns_op/R_i',
        'r_e': 'stn/stn_syns_op/R_e'
    },
    init_kwargs={'backend': 'numpy', 'solver': 'scipy', 'step_size': dt}
)
results = results * 1e3

# post-processing
#################
inputs, outputs = [], []
cutoff = 100.0

for i in range(len(etas)):

    indices_tmp = np.argwhere(np.round(result_map.loc[:, 'eta_e'].values, decimals=2) ==
                              np.round(param_grid['eta_e'][i], decimals=2)).squeeze()
    indices = result_map.index[indices_tmp]
    inputs.append(param_grid['eta_e'][i])

    outputs_tmp = []
    result_map_tmp = result_map.loc[indices, :]
    for j in range(len(ks)):
        idx_tmp = np.argwhere(np.round(result_map_tmp.loc[:, 'k_ee'].values, decimals=2) ==
                              np.round(param_grid['k_ee'][j], decimals=2)).squeeze()
        idx = result_map_tmp.index[idx_tmp]
        ts = results.loc[cutoff:, ('r_e', idx)].values
        outputs_tmp.append(np.max(ts))
    outputs.append(outputs_tmp)

plt.plot(np.asarray(inputs), np.asarray(outputs))
plt.legend([f"J = {np.round(k_0*k, decimals=1)}" for k in ks])
plt.ylabel(r'firing rate (r)')
plt.xlabel(r'background current $\mathbf{\eta}$')
plt.tight_layout()
plt.show()
