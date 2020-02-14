from pyrates.utility import grid_search, plot_connectivity
from pyrates.ir import CircuitIR
import numpy as np
import matplotlib.pyplot as plt


# parameters
dt = 1e-2
T = 10000.0
dts = 1.0

param_grid = {'eta': np.arange(-10, -2.9, 1), 'delta': np.arange(0.05, 1.5, 0.05)}
param_map = {'eta': {'nodes': ['stn'], 'vars': ['stn_simple/eta_e']},
             'delta': {'nodes': ['stn'], 'vars': ['stn_simple/delta_e']}}

# simulation
results, result_map = grid_search(circuit_template="config/stn_gpe/stn_pop",
                                  param_grid=param_grid,
                                  param_map=param_map,
                                  simulation_time=T,
                                  step_size=dt,
                                  sampling_step_size=dts,
                                  permute_grid=True,
                                  inputs={},
                                  outputs={'r_e': 'stn/stn_simple/R_e'},
                                  init_kwargs={'backend': 'numpy', 'solver': 'scipy', 'step_size': dt},
                                  )

# post-processing
results = results * 1e3
n, m = len(param_grid['eta']), len(param_grid['delta'])
rates = np.zeros((m, n))
diff = np.zeros_like(rates)
target_rate = 20.0
for id in result_map.index:
    r = results.loc[:, ('r_e', id)]
    idx_c = np.argwhere(param_grid['eta'] == result_map.at[id, 'eta'])[0]
    idx_r = np.argwhere(param_grid['delta'] == result_map.at[id, 'delta'])[0]
    rates[idx_r, idx_c] = r.iloc[-1, :]
    diff[idx_r, idx_c] = abs(r.iloc[-1, :] - target_rate)

# visualization
fig, axes = plt.subplots(ncols=2, figsize=(12, 8))
xticks = np.round(param_grid['eta'], decimals=1)
yticks = np.round(param_grid['delta'], decimals=2)
plot_connectivity(rates, ax=axes[0], xticklabels=xticks, yticklabels=yticks)
axes[0].set_xlabel('eta')
axes[0].set_ylabel('delta')
axes[0].set_title('r(STN)')
plot_connectivity(diff, ax=axes[1], xticklabels=xticks, yticklabels=yticks)
axes[1].set_xlabel('eta')
axes[1].set_ylabel('delta')
axes[1].set_title('abs(r - 20.0)')
plt.tight_layout()
plt.savefig('stn_fit.svg')
plt.show()
