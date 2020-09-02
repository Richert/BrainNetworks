import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from pyrates.utility.grid_search import grid_search
from copy import deepcopy
from scipy.ndimage.filters import gaussian_filter1d
from pyrates.utility.visualization import plot_timeseries, create_cmap
import h5py

linewidth = 1.2
fontsize1 = 10
fontsize2 = 10
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
dts = 1.0
T = 1700.0
sim_steps = int(np.round(T/dt))
stim_offset = int(np.round(0.0/dt))
stim_dur = int(np.round(500.0/dt))
stim_period = 70.0
stim_periods = [65.0]
stim_amps = [30.0]

# model parameters
k_gp = 30.0
k_p = 1.5
k_i = 0.9
k_pi = 1.0
k = 10.0
eta = 1.0
param_grid = {
        'k_ae': [100.0*k],
        'k_pe': [100.0*k],
        'k_pp': [1.0*k*k_gp*k_p/k_i],
        'k_ap': [1.0*k*k_gp*k_p*k_i*k_pi],
        'k_aa': [1.0*k*k_gp/(k_p*k_i)],
        'k_pa': [1.0*k*k_gp*k_i/(k_p*k_pi)],
        'k_ps': [200.0*k],
        'k_as': [200.0*k],
        'eta_e': [0.02],
        'eta_p': [4.8*eta],
        'eta_a': [-6.5*eta],
        'eta_s': [0.002],
        'delta_p': [90.0],
        'delta_a': [120.0],
        'tau_p': [25],
        'tau_a': [20],
        'omega': stim_periods,
        'alpha': np.asarray(stim_amps)
    }
#param_grid = pd.DataFrame.from_dict(param_grid)

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
    'omega': {'vars': ['sl_op/t_off'], 'nodes': ['driver']},
    'alpha': {'vars': ['sl_op/alpha'], 'nodes': ['driver']}
}

param_scalings = [
            #('delta_p', 'tau_p', 2.0),
            #('delta_a', 'tau_a', 2.0),
            #('k_pe', 'delta_p', 0.5),
            #('k_pp', 'delta_p', 0.5),
            #('k_pa', 'delta_p', 0.5),
            #('k_ps', 'delta_p', 0.5),
            #('k_ae', 'delta_a', 0.5),
            #('k_ap', 'delta_a', 0.5),
            #('k_aa', 'delta_a', 0.5),
            #('k_as', 'delta_a', 0.5),
            #('eta_p', 'delta_p', 1.0),
            #('eta_a', 'delta_a', 1.0),
            ]

# plotting the internal connections
conns = ['k_pp', 'k_ap', 'k_pa', 'k_aa']
connections = pd.DataFrame.from_dict({'value': [param_grid[k] for k in conns],
                                      'connection': [r'$J_{pp}$', r'$J_{ap}$', r'$J_{pa}$', r'$J_{aa}$']})
# fig, ax = plt.subplots(figsize=(3, 2), dpi=dpi)
# sns.set_color_codes("muted")
# sns.barplot(x="value", y="connection", data=connections, color="b")
# ax.set(xlim=(0, 85), ylabel="", xlabel="")
# ax.tick_params(axis='x', which='major', labelsize=9)
# sns.despine(left=True, bottom=True)
# #ax.set_title('GPe Coupling: Condition 1')
# plt.tight_layout()
# plt.show()

for key, key_tmp, power in param_scalings:
    param_grid[key] = np.asarray(param_grid[key]) * np.asarray(param_grid[key_tmp]) ** power

# ctx *= param_grid['delta_p']
# plt.plot(ctx)
# plt.show()

# simulations
#############

results, result_map = grid_search(
    circuit_template="config/stn_gpe/gpe_2pop_driver",
    param_grid=param_grid,
    param_map=param_map,
    simulation_time=T,
    step_size=dt,
    permute_grid=True,
    sampling_step_size=dts,
    inputs={
        #'driver/sl_op/alpha': inp
    },
    outputs={
        'r_i': 'gpe_p/gpe_proto_syns_op/R_i',
        'r_a': 'gpe_a/gpe_arky_syns_op/R_a'
    },
    init_kwargs={
        'backend': 'numpy', 'solver': 'scipy', 'step_size': dt},
    method='RK45',
)

results.plot()
plt.show()

fig2, ax = plt.subplots(figsize=(6, 2.0), dpi=dpi)
results = results * 1e3
plot_timeseries(results, ax=ax)
plt.legend(['GPe-p', 'GPe-a'])
ax.set_ylabel('Firing rate')
ax.set_xlabel('time (ms)')
ax.set_xlim([500.0, 1500.0])
ax.set_ylim([0.0, 120.0])
ax.tick_params(axis='both', which='major', labelsize=9)
plt.tight_layout()
plt.show()
