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
T = 500.0

# model parameters
etas = np.linspace(-10.0, 20.0, num=50)
param_grid = {
        'k_ee': [2.0],
        'eta_e': np.asarray([0.0]) + etas,
        'delta_e': [0.05],
        'tau_e': [13],
    }

param_map = {
    'k_ee': {'vars': ['weight'], 'edges': [('stn', 'stn')]},
    'eta_e': {'vars': ['stn_op/eta_e'], 'nodes': ['stn']},
    'delta_e': {'vars': ['stn_op/delta_e'], 'nodes': ['stn']},
    'tau_e': {'vars': ['stn_op/tau_e'], 'nodes': ['stn']},
}

param_scalings = [
    ('delta_e', 'tau_e', 2.0),
    ('k_ee', 'delta_e', 0.5),
    ('eta_e', 'delta_e', 1.0)
]

# grid-search
#############

for key, key_tmp, power in param_scalings:
    param_grid[key] = np.asarray(param_grid[key]) * np.asarray(param_grid[key_tmp]) ** power

for key, val in param_grid.items():
    if len(val) == 1:
        param_grid[key] = np.asarray(list(val)*len(etas))

results, result_map = grid_search(
    circuit_template="config/stn_gpe/stn_pop",
    param_grid=param_grid,
    param_map=param_map,
    simulation_time=T,
    step_size=dt,
    permute=False,
    sampling_step_size=dts,
    inputs={},
    outputs={
        'r_e': 'stn/stn_op/R_e'
    },
    init_kwargs={'backend': 'numpy', 'solver': 'scipy', 'step_size': dt},
    method='RK45'
)
results = results * 1e3

# post-processing
#################
inputs, outputs = [], []
cutoff = 400.0

for i in range(len(etas)):

    idx = result_map.index[i]
    inputs.append(result_map.loc[idx, 'eta_e'])

    ts = results.loc[cutoff:, ('r_e', idx)].values
    outputs.append(np.max(ts))

plt.plot(np.asarray(inputs), np.asarray(outputs))
plt.tight_layout()
plt.show()
