import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyrates.utility.grid_search import grid_search
import h5py
import os

# find fittest candidate among fitting results
##############################################

directory = "/home/rgast/JuliaProjects/JuRates/BasalGanglia/results/stn_gpe_nobeta_results"
fid = "stn_gpe_nobeta"
dv = 'f'

# load data into frame
df = pd.DataFrame(data=np.zeros((1, 1)), columns=["fitness"])
for fn in os.listdir(directory):
    if fn.startswith(fid) and fn.endswith(".h5"):
        f = h5py.File(f"{directory}/{fn}", 'r')
        index = int(fn.split('_')[-1][:-3])
        df_tmp = pd.DataFrame(data=np.asarray([1/f["f/f"][()]]), columns=["fitness"], index=[index])
        df = df.append(df_tmp)
df = df.iloc[1:, :]
df = df.sort_values("fitness")
print(np.mean(df.loc[:, 'fitness']))

n_fittest = 1  # increase to get the second fittest candidate and so on
fidx = df.index[-n_fittest]

# load parameter set of fittest candidate
#########################################

fname = f"{directory}/{fid}_{fidx}.h5"
dv = 'p'
ivs = ["tau_e", "tau_p", "tau_ampa_r", "tau_ampa_d", "tau_gabaa_r", "tau_gabaa_d", "tau_stn", "eta_e", "eta_p",
       "delta_e", "delta_p", "k_pe", "k_ep", "k_pp"]
f = h5py.File(fname, 'r')
data = [f[dv][key][()] for key in ivs]
param_grid = pd.DataFrame(data=np.asarray([data]), columns=ivs)

print(f"Winning parameter set: {fidx}, fitness = {df.loc[fidx, 'fitness']}")
print(param_grid.iloc[0, :])

# simulation parameter definitions
##################################

# simulation parameters
dt = 1e-3
dts = 1e-1
T = 2000.0
cutoff = 1000.0

k = 1000.0
eta = 1000.0
delta = 100.0
param_grid_final = {
        'k_ee': [0.1*param_grid.loc[0, 'k_pe']*k],
        'k_pe': [param_grid.loc[0, 'k_pe']*k],
        'k_ep': [param_grid.loc[0, 'k_ep']*k],
        'k_pp': [param_grid.loc[0, 'k_ep']*param_grid.loc[0, 'k_pp']*k],
        'eta_e': [param_grid.loc[0, 'eta_e']*eta],
        'eta_p': [param_grid.loc[0, 'eta_p']*eta],
        'delta_e': [param_grid.loc[0, 'delta_e']*delta],
        'delta_p': [param_grid.loc[0, 'delta_e']*param_grid.loc[0, 'delta_p']*delta],
        'tau_e': [param_grid.loc[0, 'tau_e']],
        'tau_p': [param_grid.loc[0, 'tau_p']],
        'tau_ampa_r': [param_grid.loc[0, 'tau_ampa_r']],
        'tau_ampa_d': [param_grid.loc[0, 'tau_ampa_d']],
        'tau_gabaa_r': [param_grid.loc[0, 'tau_gabaa_r']],
        'tau_gabaa_d': [param_grid.loc[0, 'tau_gabaa_d']],
        'tau_stn': [param_grid.loc[0, 'tau_stn']]
    }
param_grid = pd.DataFrame.from_dict(param_grid_final)

param_map = {
    'k_ee': {'vars': ['weight'], 'edges': [('stn', 'stn')]},
    'k_pe': {'vars': ['weight'], 'edges': [('stn', 'gpe_p')]},
    'k_pp': {'vars': ['weight'], 'edges': [('gpe_p', 'gpe_p')]},
    'k_ep': {'vars': ['weight'], 'edges': [('gpe_p', 'stn')]},
    'eta_e': {'vars': ['stn_syns_op/eta_e'], 'nodes': ['stn']},
    'eta_p': {'vars': ['gpe_proto_syns_op/eta_i'], 'nodes': ['gpe_p']},
    'delta_e': {'vars': ['stn_syns_op/delta_e'], 'nodes': ['stn']},
    'delta_p': {'vars': ['gpe_proto_syns_op/delta_i'], 'nodes': ['gpe_p']},
    'tau_e': {'vars': ['stn_syns_op/tau_e'], 'nodes': ['stn']},
    'tau_p': {'vars': ['gpe_proto_syns_op/tau_i'], 'nodes': ['gpe_p']},
    'tau_ampa_d': {'vars': ['gpe_proto_syns_op/tau_ampa_d', 'stn_syns_op/tau_ampa_d'], 'nodes': ['gpe_p', 'stn']},
    'tau_gabaa_d': {'vars': ['gpe_proto_syns_op/tau_gabaa_d', 'stn_syns_op/tau_gabaa_d'], 'nodes': ['gpe_p', 'stn']},
    'tau_ampa_r': {'vars': ['gpe_proto_syns_op/tau_ampa_d', 'stn_syns_op/tau_ampa_d'], 'nodes': ['gpe_p', 'stn']},
    'tau_gabaa_r': {'vars': ['gpe_proto_syns_op/tau_gabaa_d', 'stn_syns_op/tau_gabaa_d'], 'nodes': ['gpe_p', 'stn']},
    'tau_stn': {'vars': ['stn_syns_op/tau_gabaa'], 'nodes': ['stn']}
}

# simulations
#############
results, result_map = grid_search(
    circuit_template="config/stn_gpe/stn_gpe_2pop",
    param_grid=param_grid,
    param_map=param_map,
    simulation_time=T,
    step_size=dt,
    permute=True,
    sampling_step_size=dts,
    inputs={
        #'stn/stn_op/ctx': ctx,
        #'str/str_dummy_op/I': stria
        },
    outputs={'r_e': 'stn/stn_syns_op/R_e', 'r_p': 'gpe_p/gpe_proto_syns_op/R_i'},
    init_kwargs={
        'backend': 'numpy', 'solver': 'scipy', 'step_size': dt},
    method='RK45'
)

results = results.loc[cutoff:, :]*1e3
results.index = results.index*1e-3
results.plot()
#plt.xlim(0, 100)
#plt.ylim(0, 200)
plt.title(f"File-ID = {fidx}, Fitness = {df.loc[fidx, 'fitness']}")
plt.show()
