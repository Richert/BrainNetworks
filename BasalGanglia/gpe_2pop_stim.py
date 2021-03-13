import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from pyrates.utility.grid_search import grid_search
from pyrates.utility.visualization import plot_timeseries
from pyrates.utility.data_analysis import welch

linewidth = 1.2
fontsize1 = 12
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
dts = 1.0
T = 2050.0
sim_steps = int(np.round(T/dt))
stim_offset = int(np.round(0.0/dt))
stim_dur = int(np.round(500.0/dt))
stim_periods = [85.0]
stim_amps = [3.0]

# model parameters
k_gp = 1.0
k = 10.0
param_grid = {
        'k_ae': [k*1.5],
        'k_pe': [k*5.0],
        'k_pp': [1.5*k*k_gp],
        'k_ap': [2.0*k*k_gp],
        'k_aa': [0.1*k*k_gp],
        'k_pa': [5.0*k*k_gp],
        'k_ps': [k*10.0],
        'k_as': [k*1.0],
        'eta_e': [0.02],
        'eta_p': [25.0],
        'eta_a': [26.0],
        'eta_s': [0.002],
        'delta_p': [9.0],
        'delta_a': [3.0],
        'tau_p': [18],
        'tau_a': [32],
        'omega': stim_periods,
        'a2': np.asarray(stim_amps),
        'a1': [0.0]
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
    'omega': {'vars': ['sl_op/t_off'], 'nodes': ['driver']},
    'a1': {'vars': ['weight'], 'edges': [('driver', 'gpe_p', 0)]},
    'a2': {'vars': ['weight'], 'edges': [('driver', 'gpe_p', 1)]}
}

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
    method='RK45'
)

# results.plot()
# plt.show()

fig2, ax = plt.subplots(figsize=(6, 1.8), dpi=dpi)
results = results * 1e3
plot_timeseries(results, ax=ax)
plt.legend(['GPe-p', 'GPe-a'])
ax.set_ylabel('Firing rate (GPe-p)')
ax.set_xlabel('time (ms)')
ax.set_xlim([1000.0, T-50.0])
ax.set_ylim([00.0, 150.0])
ax.tick_params(axis='both', which='major', labelsize=fontsize1)
plt.tight_layout()

# fig3, ax = plt.subplots(figsize=(6, 1.8), dpi=dpi)
# results.index = results.index * 1e-3
# psds, freqs = welch(results, fmin=1.0, fmax=100.0, tmin=1.0, n_fft=2048, n_overlap=1024)
# freq_results = pd.DataFrame(data=np.log(psds.T), index=freqs, columns=['r_i'])
# plot_timeseries(freq_results, ax=ax)
# ax.set_ylabel('log PSD')
# ax.set_xlabel('frequency (Hz)')
# ax.set_ylim([-15.0, 5.0])

plt.tight_layout()
plt.show()
