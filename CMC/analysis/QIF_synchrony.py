from pyrates.utility.grid_search import grid_search
import numpy as np
import matplotlib.pyplot as plt


# Parameters
############

# general parameters
dt = np.round(5e-4, decimals=5)
dts = 1e-2
T = 1200.

# model parameters
n_samples = 200
etas = np.linspace(-6.5, -4.5, num=n_samples)
params = {'eta': etas}
param_map = {'eta': {'vars': ['Op_sd_exp/eta'], 'nodes': ['p']}
             }

# simulation
############

results, result_map = grid_search(circuit_template="model_templates.montbrio.simple_montbrio.QIF_sd_exp",
                                  param_grid=params,
                                  param_map=param_map,
                                  inputs={},
                                  outputs={"r": "p/Op_sd_exp/r", "v": "p/Op_sd_exp/v"},
                                  step_size=dt, simulation_time=T, permute_grid=True, sampling_step_size=dts,
                                  method='RK45')

# calculation of kuramoto order parameter
#########################################

eta_vals = np.zeros_like(etas)
sync_mean = np.zeros_like(etas)
sync_var = np.zeros_like(etas)
cutoff = 200.0
for i, key in enumerate(result_map.index):

    # extract data
    eta = result_map.at[key, 'eta']
    rates = results.loc[cutoff:, ('r', key)]
    potentials = results.loc[cutoff:, ('v', key)]

    # calculate order parameter (sync)
    w = np.pi*rates.values + 1.0j*potentials.values
    sync = (1 - w) / (1 + w)

    # store stuff
    eta_vals[i] = eta
    sync_mean[i] = np.mean(np.abs(sync))
    sync_var[i] = np.var(np.abs(sync))

# visualization of kuramoto order parameter
###########################################

fig, ax = plt.subplots()
ax.plot(eta_vals, sync_mean)
ax.fill_between(eta_vals, sync_mean - sync_var, sync_mean + sync_var)
ax.plot(eta_vals, sync_var)
plt.tight_layout()
plt.show()
