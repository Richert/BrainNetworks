import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyrates.utility.grid_search import grid_search
from copy import deepcopy
from scipy.ndimage.filters import gaussian_filter1d
from pyrates.utility import plot_timeseries, create_cmap
import h5py
import os

# find fittest candidate among fitting results
##############################################

directory = "/home/rgast/JuliaProjects/JuRates/BasalGanglia/results/stn_gpe_syns"
fid = "stn_gpe_syns_opt"
dv = 'f'

# load data into frame
df = pd.DataFrame(data=np.zeros((1, 1)), columns=["fitness"])
for fn in os.listdir(directory):
    if fn.startswith(fid) and fn.endswith("fitness.h5"):
        f = h5py.File(f"{directory}/{fn}", 'r')
        index = int(fn.split('_')[-2])
        df_tmp = pd.DataFrame(data=np.asarray([1/f["f"][()]]), columns=["fitness"], index=[index])
        df = df.append(df_tmp)
df = df.iloc[1:, :]
df = df.sort_values("fitness")

n_fittest = 2  # increase to get the second fittest candidate and so on
fidx = df.index[-n_fittest]

# load parameter set of fittest candidate
#########################################

fname = f"{directory}/{fid}_{fidx}_params.h5"
dv = 'p'
ivs = ['eta_e', 'eta_p', 'eta_a', 'k_ee', 'k_pe', 'k_ae', 'k_ep', 'k_pp', 'k_ap', 'k_pa', 'k_aa', 'k_ps', 'k_as',
       'delta_e', 'delta_p', 'delta_a', 'tau_e', 'tau_p', 'tau_a']
f = h5py.File(fname, 'r')
data = [f[dv][key][()] for key in ivs[:-3]] + [13.0, 25.0, 20.0]
param_grid = pd.DataFrame(data=np.asarray([data]), columns=ivs)

print(f"Winning parameter set: {fidx}, fitness = {df.loc[fidx, 'fitness']}")
print(param_grid.iloc[0, :])

# simulation parameter definitions
##################################

# simulation parameters
dt = 1e-3
dts = 1e-1
T = 600.0
sim_steps = int(np.round(T/dt))
stim_offset = int(np.round(T*0.5/dt))
stim_delayed = int(np.round((T*0.5 + 14.0)/dt))
stim_amp = 1.0
stim_var = 100.0
stim_freq = 14.0
ctx = np.zeros((sim_steps, 1))
ctx[stim_offset, 0] = stim_amp
ctx = gaussian_filter1d(ctx, stim_var, axis=0)
stria = np.zeros((sim_steps, 1))
stria[stim_delayed, 0] = stim_amp
stria = gaussian_filter1d(stria, stim_var*2.0, axis=0)
# time = np.linspace(0., T, sim_steps)
# ctx = np.sin(2.0*np.pi*stim_freq*time*1e-3)*stim_amp + stim_amp
# stria = ctx*0.005

# plt.figure()
# plt.plot(ctx)
# plt.plot(stria)
# plt.show()

# model parameters
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

# manual changes for bifurcation analysis
k = 1.0
#param_grid.loc[0, 'k_ae'] = 190.0
param_scalings = [
            ('delta_p', None, 1.0/k),
            ('delta_a', None, 1.0/k),
            ('k_ap', None, k),
            ('k_pa', None, k),
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

conditions = [{},  # healthy control -> GPe_p: 60 Hz, STN: 20 Hz, GPe_a: 30 Hz
              {'k_pe': 0.2, 'k_ae': 0.2},  # AMPA blockade in GPe -> GPe_p: 40 Hz
              {'k_ep': 0.2},  # GABAA blocker in STN -> STN: 40 Hz, GPe_p: 100 Hz
              {'k_pe': 0.2, 'k_pp': 0.2, 'k_pa': 0.2, 'k_ae': 0.2, 'k_aa': 0.2, 'k_ap': 0.2,
               'k_ps': 0.2, 'k_as': 0.2},  # AMPA blockade and GABAA blockade in GPe -> GPe_p: 70 Hz
              {'k_pp': 0.2, 'k_pa': 0.2, 'k_aa': 0.2, 'k_ap': 0.2, 'k_ps': 0.2,
               'k_as': 0.2},  # GABAA blockade in GPe -> GPe_p: 100 Hz
              #{'k_pe': 0.0, 'k_ae': 0.0},  # STN blockade -> GPe_p: 20 HZ
              #{'k_pe': 0.0, 'k_ae': 0.0, 'k_pp': 0.2, 'k_pa': 0.2, 'k_aa': 0.2, 'k_ap': 0.2,
              # 'k_ps': 0.2, 'k_as': 0.2},  # STN blockade + GABAA blockade in GPe -> GPe_p: 60 Hz
              ]

# simulations
#############

for c_dict in deepcopy(conditions):
    for key in param_grid:
        if key in c_dict:
            c_dict[key] = np.asarray(param_grid[key]) * c_dict[key]
        elif key in param_grid:
            c_dict[key] = np.asarray(param_grid[key])
    for key, key_tmp, power in param_scalings:
        c_dict[key] = c_dict[key] * c_dict[key_tmp] ** power if key_tmp else c_dict[key] * power
    param_grid_tmp = pd.DataFrame.from_dict(c_dict)
    results, result_map = grid_search(
        circuit_template="config/stn_gpe/stn_gpe_syns",
        param_grid=param_grid_tmp,
        param_map=param_map,
        simulation_time=T,
        step_size=dt,
        permute=True,
        sampling_step_size=dts,
        inputs={
            #'stn/stn_op/ctx': ctx,
            #'str/str_dummy_op/I': stria
            },
        outputs={'r_e': 'stn/stn_syns_op/R_e', 'r_i': 'gpe_p/gpe_proto_syns_op/R_i',
                 'r_a': 'gpe_a/gpe_arky_syns_op/R_a'},
        init_kwargs={
            'backend': 'numpy', 'solver': 'scipy', 'step_size': dt},
        method='RK45'
    )

    results = results*1e3
    results.plot()
    #plt.xlim(0, 100)
    #plt.ylim(0, 200)
    plt.title(f"File-ID = {fidx}, Fitness = {df.loc[fidx, 'fitness']}")
    plt.show()
