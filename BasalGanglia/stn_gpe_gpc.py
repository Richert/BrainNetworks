from pyrates.utility.grid_search import grid_search
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import correlate
import seaborn as sns

# config
########

# parameters
dt = 1e-3
dts = 1e-1
T = 2100.0
cutoff = 100.0
sim_steps = int(np.round(T/dt))

# model parameters
k_gp = 9.2
k_p = 1.5
k_i = 0.9
k_e = 1.5
k = 100.0
eta = 100.0
param_grid = {
        'k_ee': [0.6*k],
        'k_ae': [6.0*k/k_e],
        'k_pe': [6.0*k*k_e],
        'k_ep': [8.0*k],
        'k_pp': [1.0*k_gp*k_p*k/k_i],
        'k_ap': [1.0*k_gp*k_p*k_i*k],
        'k_aa': [1.0*k_gp*k/(k_p*k_i)],
        'k_pa': [1.0*k_gp*k_i*k/k_p],
        'k_ps': [20.0*k],
        'k_as': [20.0*k],
        'eta_e': [3.0*eta],
        'eta_p': [5.0*eta],
        'eta_a': [-5.0*eta],
        'eta_s': [0.002],
        'delta_e': [30.0],
        'delta_p': [90.0],
        'delta_a': [120.0],
        'tau_e': [13],
        'tau_p': [25],
        'tau_a': [20],
    }
param_grid = pd.DataFrame.from_dict(param_grid)

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
    'eta_s': {'vars': ['str_dummy_op/eta_s'], 'nodes': ['str']},
    'eta_p': {'vars': ['gpe_proto_syns_op/eta_i'], 'nodes': ['gpe_p']},
    'eta_a': {'vars': ['gpe_arky_syns_op/eta_a'], 'nodes': ['gpe_a']},
    'delta_e': {'vars': ['stn_syns_op/delta_e'], 'nodes': ['stn']},
    'delta_p': {'vars': ['gpe_proto_syns_op/delta_i'], 'nodes': ['gpe_p']},
    'delta_a': {'vars': ['gpe_arky_syns_op/delta_a'], 'nodes': ['gpe_a']},
    'tau_e': {'vars': ['stn_syns_op/tau_e'], 'nodes': ['stn']},
    'tau_p': {'vars': ['gpe_proto_syns_op/tau_i'], 'nodes': ['gpe_p']},
    'tau_a': {'vars': ['gpe_arky_syns_op/tau_a'], 'nodes': ['gpe_a']},
}

# plotting the internal connections
conns = ['k_ee', 'k_pe', 'k_ae', 'k_ep', 'k_pp', 'k_ap', 'k_pa', 'k_aa']
conn_labels = [r'$J_{ee}$', r'$J_{pe}$', r'$J_{ae}$', r'$J_{ep}$', r'$J_{pp}$', r'$J_{ap}$', r'$J_{pa}$', r'$J_{aa}$']
conn_select = [0, 1, 2, 3, 4, 5, 6, 7]
connections = pd.DataFrame.from_dict({'value': [param_grid[conns[idx]] for idx in conn_select],
                                      'connection': [conn_labels[idx] for idx in conn_select]})
fig, ax = plt.subplots(figsize=(3, 2))
sns.set_color_codes("muted")
sns.barplot(x="value", y="connection", data=connections, color="b")
ax.set(ylabel="", xlabel="")
sns.despine(left=True, bottom=True)

# grid-search
#############

results, result_map = grid_search(
    circuit_template="config/stn_gpe/stn_gpe_syns",
    param_grid=param_grid,
    param_map=param_map,
    simulation_time=T,
    step_size=dt,
    permute=True,
    sampling_step_size=dts,
    inputs={},
    outputs={
        #'r_e': 'stn/stn_syns_op/R_e',
        'r_i': 'gpe_p/gpe_proto_syns_op/R_i',
        #'r_a': 'gpe_a/gpe_arky_syns_op/R_a'
    },
    init_kwargs={
        'backend': 'numpy', 'solver': 'scipy', 'step_size': dt},
    method='RK45'
)
results = results * 1e3
results = results.loc[cutoff:, 'r_i']
results.plot()

# post-processing
#################

# normalize firing rates
ts = results.values
ts = (ts - np.mean(ts)) / np.var(ts)

# calculate the autocorrelation at globus pallidus after cortical stimulation
ac = correlate(ts, ts, mode='full')

plt.figure()
plt.plot(np.squeeze(ac))
plt.tight_layout()
plt.show()
