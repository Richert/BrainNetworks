from pyrates.utility import grid_search
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import correlate

# config
########

# parameters
dt = 1e-3
dts = 1e-1
T = 1000.0

# parameter sweeps
param_grid = {
        'k_ee': [33.8],
        'k_ae': [94.7],
        'k_pe': [397.1],
        'k_pp': [110.6],
        'k_ep': [545.6],
        'k_ap': [287.6],
        'k_aa': [122.6],
        'k_pa': [304.0],
        'k_ps': [441.8],
        'k_as': [517.7],
        'eta_e': [1.77],
        'eta_p': [-3.02],
        'eta_a': [-12.97],
        'delta_e': [0.528],
        'delta_p': [0.979],
        'delta_a': [0.991],
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
    param_grid[key] = param_grid[key] * param_grid[key_tmp] ** power
results, result_map = grid_search(
    circuit_template="config/stn_gpe/stn_gpe",
    param_grid=param_grid,
    param_map=param_map,
    simulation_time=T,
    step_size=dt,
    permute=True,
    sampling_step_size=dts,
    inputs={},
    outputs={'r_i': 'gpe_p/gpe_proto_op/R_i'},
    init_kwargs={'backend': 'numpy', 'solver': 'scipy', 'step_size': dt}
)
results = results*1e3

# post-processing
#################

# calculate power-spectral density of firing rate fluctuations and coherence between JRCs
max_freq = np.zeros((len(grid['alpha']), len(grid['omega'])))
max_pow = np.zeros_like(max_freq)
#coh = np.zeros_like(max_freq)

for key in params.index:

    val = results[('V_e', key)] - results[('V_i', key)] + results[('I', key)]
    results.insert(loc=results.shape[1], column=('V', key), value=val)

    # calculate PSDs
    freqs, power = fft(pd.DataFrame(results[('V', key)]), tmin=20.0)

    # calculate coherence
    f = params.loc[key, 'omega']
    idx = pd.IndexSlice

    #c = functional_connectivity(results.loc[idx[:], idx[['I', 'V'], key]], 'coh', fmin=f-2, fmax=f+2, faverage=True,
    #                            indices=([1], [0]), tmin=20.0, verbose=False)

    # store output quantities
    idx_c = np.argwhere(grid['omega'] == params.loc[key, 'omega'])[0]
    idx_r = np.argwhere(grid['alpha'] == params.loc[key, 'alpha'])[0]
    max_idx = np.argmax(power)
    max_freq[idx_r, idx_c] = freqs[max_idx]
    max_pow[idx_r, idx_c] = power[max_idx]
    #coh[idx_r, idx_c] = c

# visualization
###############

fig, axes = plt.subplots(ncols=2, figsize=(12, 5))

# plot dominating frequency
plot_connectivity(max_freq, xticklabels=np.round(grid['omega'], decimals=2),
                  yticklabels=np.round(grid['alpha'], decimals=4), ax=axes[0])
axes[0].set_xlabel('omega')
axes[0].set_ylabel('alpha')
axes[0].set_title('Dominant Frequency')

# plot power density of dominating frequency
plot_connectivity(max_pow, xticklabels=np.round(grid['omega'], decimals=2),
                  yticklabels=np.round(grid['alpha'], decimals=4), ax=axes[1])
axes[1].set_xlabel('omega')
axes[1].set_ylabel('alpha')
axes[1].set_title('PSD at Dominant Frequency')

# plot coherence between JRCs
# plot_connectivity(coh, xticklabels=np.round(grid['omega'], decimals=2),
#                   yticklabels=np.round(grid['alpha'], decimals=4), ax=axes[2])
# axes[2].set_xlabel('omega')
# axes[2].set_ylabel('alpha')
# axes[2].set_title('Coherence')
# plt.tight_layout()

plt.show()
