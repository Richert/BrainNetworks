import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from pyrates.utility import grid_search, plot_timeseries
from copy import deepcopy
from scipy.ndimage.filters import gaussian_filter1d
from pyrates.utility import plot_timeseries, create_cmap
import h5py

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
sns.set(style="whitegrid")

# parameter definitions
#######################

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
k_gp = 20.0
k_p = 1.0
k_i = 1.0
k_pi = 1.0
param_grid = {
        'k_ae': [100.0],
        'k_pe': [100.0],
        'k_pp': [1.0*k_gp*k_p/k_i],
        'k_ap': [1.0*k_gp*k_p*k_i*k_pi],
        'k_aa': [1.0*k_gp/(k_p*k_i)],
        'k_pa': [1.0*k_gp*k_i/(k_p*k_pi)],
        'k_ps': [200.0],
        'k_as': [200.0],
        'eta_e': [0.02],
        'eta_p': [4.0],
        'eta_a': [0.0],
        'eta_s': [0.002],
        'delta_p': [0.1],
        'delta_a': [0.2],
        'tau_p': [25],
        'tau_a': [20],
    }
param_grid = pd.DataFrame.from_dict(param_grid)

# fname = "/home/rgast/JuliaProjects/JuRates/BasalGanglia/results/stn_gpe_ev_opt_results_final/stn_gpe_ev_opt2_41_params.h5"
# dv = 'p'
# ivs = ['eta_e', 'eta_p', 'eta_a', 'k_ee', 'k_pe', 'k_ae', 'k_ep', 'k_pp', 'k_ap', 'k_pa', 'k_aa', 'k_ps', 'k_as',
#        'delta_e', 'delta_p', 'delta_a', 'tau_e', 'tau_p', 'tau_a']
# f = h5py.File(fname, 'r')
# data = [f[dv][key][()] for key in ivs[:-3]] + [13.0, 25.0, 20.0]
# param_grid = pd.DataFrame(data=np.asarray([data]), columns=ivs)

param_map = {
    'k_ae': {'vars': ['weight'], 'edges': [('stn', 'gpe_a')]},
    'k_pe': {'vars': ['weight'], 'edges': [('stn', 'gpe_p')]},
    'k_pp': {'vars': ['weight'], 'edges': [('gpe_p', 'gpe_p')]},
    'k_ap': {'vars': ['weight'], 'edges': [('gpe_p', 'gpe_a')]},
    'k_aa': {'vars': ['weight'], 'edges': [('gpe_a', 'gpe_a')]},
    'k_pa': {'vars': ['weight'], 'edges': [('gpe_a', 'gpe_p')]},
    'k_ps': {'vars': ['weight'], 'edges': [('str', 'gpe_p')]},
    'k_as': {'vars': ['weight'], 'edges': [('str', 'gpe_a')]},
    'eta_p': {'vars': ['gpe_proto_syns_op/eta_i'], 'nodes': ['gpe_p']},
    'eta_a': {'vars': ['gpe_arky_syns_op/eta_a'], 'nodes': ['gpe_a']},
    'eta_e': {'vars': ['stn_dummy_op/eta_e'], 'nodes': ['stn']},
    'eta_s': {'vars': ['str_dummy_op/eta_s'], 'nodes': ['str']},
    'delta_p': {'vars': ['gpe_proto_syns_op/delta_i'], 'nodes': ['gpe_p']},
    'delta_a': {'vars': ['gpe_arky_syns_op/delta_a'], 'nodes': ['gpe_a']},
    'tau_p': {'vars': ['gpe_proto_syns_op/tau_i'], 'nodes': ['gpe_p']},
    'tau_a': {'vars': ['gpe_arky_syns_op/tau_a'], 'nodes': ['gpe_a']},
}

# manual changes for bifurcation analysis
#param_grid.loc[0, 'k_ae'] = 190.0

param_scalings = [
            #('delta_p', None, 1.0/k),
            #('delta_a', None, 1.0/k),
            #('k_ap', None, k),
            #('k_pa', None, k),
            ('delta_p', 'tau_p', 2.0),
            ('delta_a', 'tau_a', 2.0),
            ('k_pe', 'delta_p', 0.5),
            ('k_pp', 'delta_p', 0.5),
            ('k_pa', 'delta_p', 0.5),
            ('k_ps', 'delta_p', 0.5),
            ('k_ae', 'delta_a', 0.5),
            ('k_ap', 'delta_a', 0.5),
            ('k_aa', 'delta_a', 0.5),
            ('k_as', 'delta_a', 0.5),
            ('eta_p', 'delta_p', 1.0),
            ('eta_a', 'delta_a', 1.0),
            ]

conditions = [{},  # healthy control -> GPe_p: 60 Hz, STN: 20 Hz, GPe_a: 30 Hz
              {'k_pe': 0.0, 'k_ae': 0.0},  # AMPA blockade in GPe -> GPe_p: 40 Hz
              {'k_pe': 0.0, 'k_pp': 0.2, 'k_pa': 0.2, 'k_ae': 0.0, 'k_aa': 0.2, 'k_ap': 0.2,
               'k_ps': 0.2, 'k_as': 0.2},  # AMPA blockade and GABAA blockade in GPe -> GPe_p: 70 Hz
              {'k_pp': 0.2, 'k_pa': 0.2, 'k_aa': 0.2, 'k_ap': 0.2, 'k_ps': 0.2,
               'k_as': 0.2},  # GABAA blockade in GPe -> GPe_p: 100 Hz
              ]

# plotting the internal connections
conns = ['k_pp', 'k_ap', 'k_pa', 'k_aa']
connections = pd.DataFrame.from_dict({'value': [param_grid[k] for k in conns],
                                      'connection': [r'$k_{pp}$', r'$k_{ap}$', r'$k_{pa}$', r'$k_{aa}$']})
fig, ax = plt.subplots(figsize=(4, 3))
sns.set_color_codes("muted")
sns.barplot(x="value", y="connection", data=connections, color="b")
ax.set(xlim=(0, 85), ylabel="", xlabel="")
sns.despine(left=True, bottom=True)
#ax.set_title('GPe Coupling: Condition 1')
plt.show()

# simulations
#############

for c_dict in deepcopy(conditions):
    for key in param_grid:
        if key in c_dict:
            c_dict[key] = np.asarray(param_grid[key]) * c_dict[key]
        elif key in param_grid:
            c_dict[key] = np.asarray(param_grid[key])
    for key, key_tmp, power in param_scalings:
        c_dict[key] = c_dict[key] * c_dict[key_tmp] ** power
    param_grid_tmp = pd.DataFrame.from_dict(c_dict)
    results, result_map = grid_search(
        circuit_template="config/stn_gpe/gpe_2pop",
        param_grid=param_grid_tmp,
        param_map=param_map,
        simulation_time=T,
        step_size=dt,
        permute_grid=True,
        sampling_step_size=dts,
        inputs={
            #'stn/stn_op/ctx': ctx,
            #'str/str_dummy_op/I': stria
            },
        outputs={'r_i': 'gpe_p/gpe_proto_syns_op/R_i',
                 'r_a': 'gpe_a/gpe_arky_syns_op/R_a'},
        init_kwargs={
            'backend': 'numpy', 'solver': 'scipy', 'step_size': dt},
        method='RK45'
    )

    results = results*1e3
    fig, ax = plt.subplots(figsize=(4, 3))
    plot_timeseries(results, ax=ax)
    plt.legend(['GPe-p', 'GPe-a'])
    plt.show()
