import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from pyrates.utility import grid_search, welch, create_cmap, plot_connectivity
from scipy.interpolate.interpolate import interp1d

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
dt = 1e-2
dts = 1e-1
T = 6000.0

# stimulation parameters
stim_freqs = np.linspace(4, 20, 10)
n_infreqs = len(stim_freqs)

# model parameters
k_gp = 20.0
k_p = 1.0
k_i = 0.5
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
        'eta_p': [3.0],
        'eta_a': [-5.0],
        'eta_s': [0.002],
        'delta_p': [0.1],
        'delta_a': [0.2],
        'tau_p': [25],
        'tau_a': [20],
        'omega': np.asarray(stim_freqs),
        'w': [200.0],
        'alpha': 0.9 + np.asarray([np.min([0.09, 2.0/f**2]) for f in stim_freqs])
    }

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
    'omega': {'vars': ['sl_op/omega'], 'nodes': ['driver']},
    'alpha': {'vars': ['sl_op/alpha'], 'nodes': ['driver']},
    'w': {'vars': ['weight'], 'edges': [('driver', 'gpe_a', 0), ('driver', 'gpe_a', 1)]}
}

param_scalings = [
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
    ('eta_a', 'delta_a', 1.0)
            ]

for key, val in param_grid.items():
    if len(val) == 1:
        param_grid[key] = val * n_infreqs
for key, key_tmp, power in param_scalings:
    param_grid[key] = np.asarray(param_grid[key]) * np.asarray(param_grid[key_tmp]) ** power

# simulations
#############

results, result_map = grid_search(
    circuit_template="config/stn_gpe/gpe_2pop_driver",
    param_grid=param_grid,
    param_map=param_map,
    simulation_time=T,
    step_size=dt,
    permute_grid=False,
    sampling_step_size=dts,
    inputs={},
    outputs={'r_i': 'gpe_p/gpe_proto_syns_op/R_i',
             'r_a': 'gpe_a/gpe_arky_syns_op/R_a',
             'd': 'driver/sl_op/Z1'},
    init_kwargs={
        'backend': 'numpy', 'solver': 'scipy', 'step_size': dt, 'matrix_sparseness': 1.0},
    method='RK45'
)

# coherence calculation
#######################

results = results.loc[1000.0:, :]
results.index *= 1e-3
results = results * 1e3
powers = []
freq_labels = []
target_freqs = np.linspace(4.0, 80.0, 50)
for id in result_map.index:
    r = results.loc[:, ('r_i', id)]
    freq_labels.append(result_map.at[id, 'omega'])
    psds, freqs = welch(r, fmin=1.0, fmax=100.0, n_fft=8192, n_overlap=512)
    freq_inter = interp1d(freqs, psds.squeeze(), kind='cubic', axis=0)
    powers.append(freq_inter(target_freqs))

# plotting
##########

# coherences
cmap = create_cmap("pyrates_blue", n_colors=64, as_cmap=False, reverse=True)
powers = np.asarray(powers)
powers[powers < 0] = 0.0
ax = plot_connectivity(np.sqrt(powers), cmap=cmap)
ax.set_xticks(ax.get_xticks()[0::8])
ax.set_yticks(ax.get_yticks()[0::2])
ax.set_xticklabels(np.round(target_freqs, decimals=1)[0::8], rotation='horizontal')
ax.set_yticklabels(np.round(freq_labels, decimals=1)[0::2], rotation='horizontal')
ax.set_xlabel('omega')
ax.set_ylabel('alpha')
plt.tight_layout()

# timeseries
#results.loc[:, 'r_i'].plot()

plt.show()
