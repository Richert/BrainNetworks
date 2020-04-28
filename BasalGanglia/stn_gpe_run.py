from pyrates.utility import plot_timeseries, create_cmap, grid_search
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import correlate
import matplotlib as mpl

plt.style.reload_library()
plt.style.use('ggplot')
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.labelcolor'] = 'black'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['ytick.color'] = 'black'
mpl.rcParams['legend.fontsize'] = 12

# simulation parameters
dt = 1e-3
dts = 1e-1
T = 1000.0
sim_steps = int(np.round(T/dt))
stim_offset = int(np.round(T*0.5/dt))
stim_delayed = int(np.round((T*0.5 + 14.0)/dt))
stim_amp = 1.0
stim_var = 100.0
stim_freq = 14.0
# ctx = np.zeros((sim_steps, 1))
# ctx[stim_offset, 0] = stim_amplitude
# ctx = gaussian_filter1d(ctx, stim_var, axis=0)
# stria = np.zeros((sim_steps, 1))
# stria[stim_delayed, 0] = stim_amplitude
# stria = gaussian_filter1d(stria, stim_var*2.0, axis=0)
time = np.linspace(0., T, sim_steps)
ctx = np.sin(2.0*np.pi*stim_freq*time*1e-3)*stim_amp + stim_amp
stria = ctx*0.5

# model parameters
k = 1.0
param_grid = {
        'k_ee': [26.2],
        'k_ae': [82.2],
        'k_pe': [352.7],
        'k_pp': [54.2],
        'k_ep': [355.6],
        'k_ap': [107.4*k],
        'k_aa': [128.8],
        'k_pa': [154.2],
        'k_ps': [234.1],
        'k_as': [232.4],
        'eta_e': [-1.79],
        'eta_p': [-4.78],
        'eta_a': [-10.50],
        'delta_e': [0.463],
        'delta_p': [0.757/k],
        'delta_a': [0.768/k],
        'tau_e': [13],
        'tau_p': [25],
        'tau_a': [20],
    }

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

param_scalings = [
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
    param_grid[key] = np.asarray(param_grid[key]) * np.asarray(param_grid[key_tmp]) ** power
results, result_map = grid_search(
    circuit_template="config/stn_gpe/stn_gpe",
    param_grid=param_grid,
    param_map=param_map,
    simulation_time=T,
    step_size=dt,
    permute=True,
    sampling_step_size=dts,
    inputs={'stn/stn_op/ctx': ctx, 'str/str_dummy_op/I': stria},
    outputs={'r_i': 'gpe_p/gpe_proto_op/R_i', 'r_e': 'stn/stn_op/R_e'},
    init_kwargs={'backend': 'numpy', 'solver': 'scipy', 'step_size': dt}
)
results = results * 1e3
results = results.loc[0.5*T:, :]
results.plot()
plt.show()

# post-processing
#################
ac = []

for d, k in zip(param_grid['delta_p'], param_grid['k_ae']):

    # get index of simulation with parameters d and k
    idx = result_map.loc[(result_map.loc[:, ('delta_p', 'k_ae')] == (d, k)).all(1), :].index

    # add white noise to timeseries
    ts = results.loc[:, ('r_i', idx)].values
    ts = (ts - np.mean(ts)) / np.var(ts)

    # calculate the autocorrelation at globus pallidus after cortical stimulation
    ac.append(correlate(ts, ts, mode='same'))

plt.plot(np.squeeze(ac))
plt.tight_layout()
plt.show()
