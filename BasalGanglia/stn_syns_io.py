from pyrates.utility import plot_timeseries, create_cmap, grid_search
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import correlate
import matplotlib as mpl

print(mpl.get_backend())
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
etas = np.linspace(-8.0, 8.0, num=40)
ks = np.asarray([0.2, 0.6, 1.0, 1.4, 1.8])
k_0 = 4.0
param_grid = {
        'k_ee': np.asarray([k_0])*ks,
        'eta_e': np.asarray([0.0]) + etas,
        'delta_e': np.asarray([0.04]),
        'tau_e': np.asarray([13]),
    }

param_map = {
    'k_ee': {'vars': ['weight'], 'edges': [('stn', 'stn')]},
    'eta_e': {'vars': ['stn_syns_op/eta_e'], 'nodes': ['stn']},
    'delta_e': {'vars': ['stn_syns_op/delta_e'], 'nodes': ['stn']},
    'tau_e': {'vars': ['stn_syns_op/tau_e'], 'nodes': ['stn']},
}

param_scalings = [
    ('delta_e', 'tau_e', 2.0),
    ('k_ee', 'delta_e', 0.5),
    ('eta_e', 'delta_e', 1.0)
]

# grid-search
#############

for key, key_tmp, power in param_scalings:
    param_grid[key] = param_grid[key] * param_grid[key_tmp] ** power

#for key, val in param_grid.items():
#    if len(val) == 1:
#        param_grid[key] = np.asarray(list(val)*len(etas))

results, result_map = grid_search(
    circuit_template="config/stn_gpe/stn_syns_pop",
    param_grid=param_grid,
    param_map=param_map,
    simulation_time=T,
    step_size=dt,
    permute_grid=True,
    sampling_step_size=dts,
    inputs={},
    outputs={
        'r_e': 'stn/stn_syns_op/R_e'
    },
    init_kwargs={'backend': 'numpy', 'solver': 'scipy', 'step_size': dt},
    method='RK45'
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
