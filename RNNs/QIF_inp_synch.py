from pyrates.utility.grid_search import grid_search
from pyrates.utility.visualization import plot_connectivity
import numpy as np
import matplotlib.pyplot as plt


# Parameters
############

# general parameters
dt = 1e-4
dts = 1e-2
T = 400.

# model parameters
n_etas = 100
n_alphas = 5
etas = np.linspace(-6.5, -4.5, num=n_etas)
alphas = np.linspace(0.0, 0.2, num=n_alphas)
params = {'eta': etas, 'alpha': alphas}
param_map = {'eta': {'vars': ['Op_sd_exp/eta'], 'nodes': ['qif']},
             'alpha': {'vars': ['weight'], 'edges': [('inp', 'qif')]}
             }

# simulation
############

results, result_map = grid_search(circuit_template="qifs/QIF_sd_lorenz",
                                  param_grid=params,
                                  param_map=param_map,
                                  inputs={},
                                  outputs={"r": "qif/Op_sd_exp/r", "v": "qif/Op_sd_exp/v"},
                                  step_size=dt, simulation_time=T, permute_grid=True, sampling_step_size=dts,
                                  method='RK45')

# calculation of kuramoto order parameter
#########################################

sync_mean_2D = np.zeros((n_alphas, n_etas))
sync_var_2D = np.zeros_like(sync_mean_2D)
cutoff = 200.0
for i, key in enumerate(result_map.index):

    # extract data
    eta = result_map.at[key, 'eta']
    alpha = result_map.at[key, 'alpha']
    rates = results.loc[cutoff:, ('r', key)]
    potentials = results.loc[cutoff:, ('v', key)]

    # calculate order parameter (sync)
    w = np.pi*rates.values + 1.0j*potentials.values
    sync = (1 - w) / (1 + w)

    # store results in 2D arrays
    col = np.argmin(np.abs(etas - eta))
    row = np.argmin(np.abs(alphas - alpha))
    sync_mean_2D[row, col] = np.mean(np.abs(sync))
    sync_var_2D[row, col] = np.var(np.abs(sync))

# visualization of kuramoto order parameter
###########################################

# 2D
fig2, ax2 = plt.subplots(ncols=2, figsize=(14, 6))
ax = ax2[0]
ax = plot_connectivity(sync_mean_2D, yticklabels=np.round(alphas, decimals=1), xticklabels=np.round(etas, decimals=1),
                       ax=ax)
ax.set_title('mean of synchrony')
ax = ax2[1]
ax = plot_connectivity(sync_var_2D, yticklabels=np.round(alphas, decimals=1), xticklabels=np.round(etas, decimals=1),
                       ax=ax)
ax.set_title('var of synchrony')
plt.tight_layout()

plt.show()
