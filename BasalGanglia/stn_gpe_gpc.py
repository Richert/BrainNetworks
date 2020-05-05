from pyrates.utility import grid_search
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import correlate
from scipy.ndimage.filters import gaussian_filter1d
import h5py

# config
########

# parameters
dt = 1e-3
dts = 1e-2
T = 600.0
sim_steps = int(np.round(T/dt))
stim_offset = int(np.round(T*0.5/dt))
stim_dur = int(np.round(4.0/dt))
stim_delayed = int(np.round((T*0.5 + 10.0)/dt))
stim_amp = 4.0
stim_var = int(np.round(0.5/dt))
stim_freq = 14.0
ctx = np.zeros((sim_steps, 1))
ctx[stim_offset:stim_offset+stim_dur, 0] = stim_amp
ctx = gaussian_filter1d(ctx, stim_var, axis=0)
stria = np.zeros((sim_steps, 1))
stria[stim_delayed:stim_delayed+stim_dur, 0] = stim_amp
stria = gaussian_filter1d(stria, stim_var*2.0, axis=0)
# time = np.linspace(0., T, sim_steps)
# ctx = np.sin(2.0*np.pi*stim_freq*time*1e-3)*stim_amp + stim_amp
# stria = ctx*0.005

plt.figure()
plt.plot(ctx)
plt.plot(stria)
plt.show()

# model parameters
k = 1.0

param_grid = {
        'k_ee': [3.2],
        'k_ae': [80.5],
        'k_pe': [68.2],
        'k_pp': [4.3],
        'k_ep': [10.8],
        'k_ap': [17.0],
        'k_aa': [0.8],
        'k_pa': [29.3],
        'k_ps': [169.5],
        'k_as': [219.3],
        'eta_e': [-0.12],
        'eta_p': [-0.19],
        'eta_a': [-3.02],
        'delta_e': [0.045],
        'delta_p': [0.152],
        'delta_a': [0.126],
        'tau_e': [13],
        'tau_p': [25],
        'tau_a': [20],
    }
param_grid = pd.DataFrame.from_dict(param_grid)

# directory = "/home/rgast/JuliaProjects/JuRates/BasalGanglia/results/stn_gpe_ev_opt_results_final"
# fname = "stn_gpe_ev_opt_79_params.h5"
# dv = 'p'
# ivs = ['eta_e', 'eta_p', 'eta_a', 'k_ee', 'k_pe', 'k_ae', 'k_ep', 'k_pp', 'k_ap', 'k_pa', 'k_aa', 'k_ps', 'k_as',
#        'delta_e', 'delta_p', 'delta_a', 'tau_e', 'tau_p', 'tau_a']
# f = h5py.File(f"{directory}/{fname}", 'r')
# data = [f[dv][key][()] for key in ivs[:-3]] + [13.0, 25.0, 20.0]
# param_grid = pd.DataFrame(data=np.asarray([data]), columns=ivs)

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

# manual changes for bifurcation analysis
#param_grid.loc[0, 'eta_e'] = 3.5

param_scalings = [
            #('delta_p', None, 1.0/k),
            #('delta_a', None, 1.0/k),
            #('k_ap', None, k),
            #('k_pa', None, k),
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
    param_grid[key] = np.asarray(param_grid[key]) * np.asarray(param_grid[key_tmp]) ** power if key_tmp \
        else np.asarray(param_grid[key]) * power
results, result_map = grid_search(
        circuit_template="config/stn_gpe/stn_gpe",
        param_grid=param_grid,
        param_map=param_map,
        simulation_time=T,
        step_size=dt,
        permute=True,
        sampling_step_size=dts,
        inputs={
            'stn/stn_op/ctx': ctx,
            'str/str_dummy_op/I': stria
            },
        outputs={'r_e': 'stn/stn_op/R_e', 'r_i': 'gpe_p/gpe_proto_op/R_i', 'r_a': 'gpe_a/gpe_arky_op/R_a'},
        init_kwargs={
            'backend': 'numpy', 'solver': 'scipy', 'step_size': dt},
        method='RK45',
        max_step=0.1
    )
results = results * 1e3
#results = results.loc[0.5 * T:, :]
results.plot()

# post-processing
#################

# add white noise to timeseries
ts = results.loc[:, 'r_i'].values + np.random.normal(0.0, 1.0, size=results.shape)
ts = (ts - np.mean(ts)) / np.var(ts)

# calculate the autocorrelation at globus pallidus after cortical stimulation
ac = correlate(ts, ts, mode='same')

plt.figure()
plt.plot(np.squeeze(ac))
plt.tight_layout()
plt.show()
