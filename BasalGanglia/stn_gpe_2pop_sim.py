import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyrates.utility.grid_search import grid_search
from copy import deepcopy
import seaborn as sns
import matplotlib as mpl

linewidth = 1.2
fontsize1 = 10
fontsize2 = 12
markersize1 = 60
markersize2 = 60
dpi = 200

plt.style.reload_library()
plt.style.use('seaborn-whitegrid')
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']
#mpl.rc('text', usetex=True)
mpl.rcParams["font.sans-serif"] = ["Roboto"]
mpl.rcParams["font.size"] = fontsize1
mpl.rcParams["font.weight"] = "bold"
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rcParams['axes.titlesize'] = fontsize2
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['axes.labelsize'] = 'large'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['ytick.color'] = 'black'
mpl.rcParams['ytick.alignment'] = 'center'
mpl.rcParams['legend.fontsize'] = fontsize1

# parameter definitions
#######################

# simulation parameters
dt = 1e-4
dts = 1e-1
T = 2050.0

# model parameters
k = 10.0
param_grid = {
        'k_ee': [0.4*k],
        'k_pe': [5.0*k],
        'k_ep': [1.5*k],
        'k_pp': [1.0*k],
        'eta_e': [12.0],
        'eta_p': [2.0],
        'delta_e': [2.0],
        'delta_p': [10.0],
        'tau_e': [13.0],
        'tau_p': [19.0],
        'tau_ampa_r': [0.8],
        'tau_ampa_d': [3.7],
        'tau_gabaa_r': [0.5],
        'tau_gabaa_d': [5.0],
        'tau_stn': [2.0]
    }
param_grid = pd.DataFrame.from_dict(param_grid)

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
    'tau_ampa_r': {'vars': ['gpe_proto_syns_op/tau_ampa_r', 'stn_syns_op/tau_ampa_r'], 'nodes': ['gpe_p', 'stn']},
    'tau_ampa_d': {'vars': ['gpe_proto_syns_op/tau_ampa_d', 'stn_syns_op/tau_ampa_d'], 'nodes': ['gpe_p', 'stn']},
    'tau_gabaa_r': {'vars': ['gpe_proto_syns_op/tau_gabaa_r', 'stn_syns_op/tau_gabaa_r'], 'nodes': ['gpe_p', 'stn']},
    'tau_gabaa_d': {'vars': ['gpe_proto_syns_op/tau_gabaa_d', 'stn_syns_op/tau_gabaa_d'], 'nodes': ['gpe_p', 'stn']},
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

results = results*1e3
results.plot()
plt.show()
