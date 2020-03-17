from pyrates.utility import plot_timeseries, grid_search, plot_psd, plot_connectivity, create_cmap
import numpy as np
import matplotlib.pyplot as plt

__author__ = "Richard Gast"
__status__ = "Development"


# parameters
dt = 1e-5
T = 5.
inp = 3. + np.random.randn(int(T/dt), 1) * 0.0
inp[0:int(1/dt)] = 0.

params = {'H_e': np.arange(1.0, 2.0, 0.5)[::-1], 'H_i': np.arange(0.001, 0.005, 0.001)}
param_map = {'H_e': {'var': [('Op_e.0', 'beta'), ('Op_i.0', 'beta')],
                     'nodes': ['E.0', 'I.0']},
             'H_i': {'var': [('Op_e.0', 'alpha'), ('Op_i.0', 'alpha')],
                     'nodes': ['E.0', 'I.0']},
             }

# perform simulation
results = grid_search(circuit_template="../config/cmc_templates.EI",
                      param_grid=params, param_map=param_map,
                      inputs={("E.0", "Op_e.0", "i_in"): inp}, outputs={"r": ("E.0", "Op_e.0", "r")},
                      dt=dt, simulation_time=T, permute_grid=True, sampling_step_size=1e-3)

# plot raw timeseries
for condition in results.columns.values:
    ax = plot_timeseries(results[condition[0]][condition[1]])
    title = f'{results.columns.names[0]} = {condition[0]}, {results.columns.names[1]} = {condition[1]}'
    ax.set_title(condition)
    plt.show()

cm1 = create_cmap('pyrates_red', as_cmap=True, n_colors=16)
cm2 = create_cmap('pyrates_green', as_cmap=True, n_colors=16)
cm3 = create_cmap('pyrates_blue/pyrates_red', as_cmap=True, n_colors=16, pyrates_blue={'reverse': True})

for idx in params['H_e']:
    plot_psd(results[idx], cmap=cm3)
    plt.show()

# get maximum frequency and power at that frequency
cut_off = 1.
max_freq = np.zeros((len(params['H_e']), len(params['H_i'])))
freq_pow = np.zeros_like(max_freq)
for i, H_e in enumerate(params['H_e']):
    for j, H_i in enumerate(params['H_i']):
        if not results[H_e][H_i].isnull().any().any():
            _ = plot_psd(results[H_e][H_i], tmin=cut_off, show=False)
            pow = plt.gca().get_lines()[-1].get_ydata()
            freqs = plt.gca().get_lines()[-1].get_xdata()
            max_freq[i, j] = freqs[np.argmax(pow)]
            freq_pow[i, j] = np.max(pow)
            plt.close('all')

# plot maximum response frequency
fig, ax = plt.subplots(ncols=2, figsize=(15, 5), gridspec_kw={})
cax1 = plot_connectivity(max_freq, ax=ax[0], yticklabels=list(np.round(params['H_e'], decimals=2)),
                         xticklabels=list(np.round(params['H_i'], decimals=2)), cmap=cm1)
cax1.set_xlabel('H_i')
cax1.set_ylabel('H_e')
cax1.set_title(f'max freq')
cax2 = plot_connectivity(freq_pow, ax=ax[1], yticklabels=list(np.round(params['H_e'], decimals=2)),
                         xticklabels=list(np.round(params['H_i'], decimals=2)), cmap=cm2)
cax2.set_xlabel('H_i')
cax2.set_ylabel('H_e')
cax2.set_title(f'freq pow')
plt.suptitle('EI-circuit sensitivity to synaptic efficacies (H)')
plt.tight_layout(pad=2.5, rect=(0.01, 0.01, 0.99, 0.96))
#fig.savefig("/home/rgast/Documents/Studium/PhD_Leipzig/Figures/BGTCS/eic_efficacies", format="svg")
plt.show()
