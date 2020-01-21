from pyrates.utility import fft, plot_connectivity, ClusterGridSearch
import matplotlib.pyplot as plt
import numpy as np
import os
from pandas import DataFrame, read_hdf

# define system parameters
param_map = {
        'd': {'vars': ['delay'], 'edges': [('stn', 'gpe'), ('gpe', 'stn')]},
        's': {'vars': ['spread'], 'edges': [('stn', 'gpe'), ('gpe', 'stn')]}
    }

param_grid = {
        'd': np.arange(0, 8.0, 1.0),
        's': np.arange(0.5, 2.2, 0.25)
    }

# define simulation parameters
dt = 1e-3
T = 5000.0
dts = 2e-1
nodes = ['animals']
chunk_sizes = [50]
compute_dir = f"{os.getcwd()}/stn_gpe_delay_analysis"

# perform simulation
cgs = ClusterGridSearch(nodes=nodes, compute_dir=compute_dir)
fname = cgs.run(circuit_template=f"{os.getcwd()}/config/stn_gpe/net_qif_syn_adapt",
                param_grid=param_grid,
                param_map=param_map,
                simulation_time=T,
                dt=dt,
                sampling_step_size=dts,
                permute_grid=True,
                inputs={},
                outputs={'r': "gpe/qif_gpe/R_i"},
                chunk_size=chunk_sizes,
                worker_env="/nobackup/spanien1/rgast/anaconda3/envs/pyrates_test/bin/python3",
                worker_file=f'{os.getcwd()}/stn_gpe_worker_delays.py',
                gs_kwargs={'init_kwargs': {'backend': 'numpy', 'solver': 'scipy', 'step_size': dt}},
                )
#fname = "/nobackup/spanien1/rgast/PycharmProjects/BrainNetworks/BasalGanglia/stn_gpe_delay_analysis/Results/DefaultGrid_8/CGS_result_DefaultGrid_8.h5"
results = read_hdf(fname, key='Results/results')
params = read_hdf(fname, key="Results/result_map")

# extract spectral properties
max_freq = np.zeros((len(param_grid['s']), len(param_grid['d'])))
max_pow = np.zeros_like(max_freq)

for key in params.index:

    # calculate PSDs
    freqs, power = fft(DataFrame(results[('r', key)]), tmin=0.5)

    # store output quantities
    idx_c = np.argwhere(param_grid['d'] == params.loc[key, 'd'])[0]
    idx_r = np.argwhere(param_grid['s'] == params.loc[key, 's'])[0]
    max_idx = np.argmax(power)
    max_freq[idx_r, idx_c] = freqs[max_idx]
    max_pow[idx_r, idx_c] = power[max_idx]

# visualization
###############
results.plot()

fig, axes = plt.subplots(ncols=2, figsize=(12, 5))

# plot dominating frequency
plot_connectivity(max_freq, xticklabels=np.round(param_grid['d'], decimals=2),
                  yticklabels=np.round(param_grid['s'], decimals=3), ax=axes[0])
axes[0].set_xlabel('delay mean')
axes[0].set_ylabel('delay std')
axes[0].set_title('Dominant Frequency')

# plot power density of dominating frequency
plot_connectivity(max_pow, xticklabels=np.round(param_grid['d'], decimals=2),
                  yticklabels=np.round(param_grid['s'], decimals=3), ax=axes[1])
axes[1].set_xlabel('delay mean')
axes[1].set_ylabel('delay std')
axes[1].set_title('PSD at Dominant Frequency')

plt.savefig(f"{compute_dir}/oscillations", dpi=600, format='svg')
plt.show()
