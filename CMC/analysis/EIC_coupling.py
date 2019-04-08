from pyrates.utility import plot_timeseries, grid_search, plot_psd, plot_connectivity
import numpy as np
import matplotlib.pyplot as plt
from seaborn import cubehelix_palette
from scipy.signal import find_peaks

__author__ = "Richard Gast"
__status__ = "Development"


# parameters
dt = 1e-4
dts = 1e-3
T = 50.
cut_off = int(10.0/dts)
Js = [10., 15., 20.]
ei_ratio = np.arange(1.0, 10.1, 1.0)[::-1]
io_ratio = np.arange(1.0, 10.1, 1.0)
J_ei = np.zeros((int(len(ei_ratio) * len(io_ratio))))
J_ie = np.zeros_like(J_ei)
J_ee = np.zeros_like(J_ei)
J_ii = np.zeros_like(J_ei)

fig, ax = plt.subplots(ncols=len(Js), nrows=2, figsize=(20, 15), gridspec_kw={})
for idx, J in enumerate(Js):

    J_ee[:] = J
    n = 0
    for r_ei in ei_ratio:
        for r_io in io_ratio:
            J_ii[n] = J / r_ei
            J_ie[n] = J/ r_io
            J_ei[n] = J / (r_ei * r_io)
            n += 1

    params = {'J_ee': J_ee, 'J_ii': J_ii, 'J_ie': J_ie, 'J_ei': J_ei}
    param_map = {'J_ee': {'var': [('Op_e_adapt.0', 'J')],
                          'nodes': ['E.0']},
                 'J_ii': {'var': [('Op_i_adapt.0', 'J')],
                          'nodes': ['I.0']},
                 'J_ei': {'var': [(None, 'weight')],
                          'edges': [('I.0', 'E.0', 0)]},
                 'J_ie': {'var': [(None, 'weight')],
                          'edges': [('E.0', 'I.0', 0)]}
                 }

    # perform simulation
    results = grid_search(circuit_template="../config/cmc_templates.EI_adapt",
                          param_grid=params, param_map=param_map,
                          inputs={}, outputs={"r_e": ("E.0", "Op_e_adapt.0", "r")},
                          dt=dt, simulation_time=T, permute_grid=False, sampling_step_size=dts)

    # plotting
    max_freq = np.zeros((len(ei_ratio), len(io_ratio)))
    freq_pow = np.zeros_like(max_freq)
    for j_ee, j_ii, j_ie, j_ei in zip(params['J_ee'], params['J_ii'], params['J_ie'], params['J_ei']):
        data = results[j_ee][j_ii][j_ie][j_ei].values[cut_off:, 0]
        peaks, _ = find_peaks(data, distance=int(1./dts))
        r, c = np.argmin(np.abs(ei_ratio - j_ee/j_ii)), np.argmin(np.abs(io_ratio - j_ee/j_ie))
        if len(peaks) > 0:
            max_freq[r, c] = T/len(peaks)
            freq_pow[r, c] = np.mean(data[peaks])

    cm1 = cubehelix_palette(n_colors=int(len(ei_ratio)*len(io_ratio)), as_cmap=True, start=2.5, rot=-0.1)
    cm2 = cubehelix_palette(n_colors=int(len(ei_ratio)*len(io_ratio)), as_cmap=True, start=-2.0, rot=-0.1)
    cax1 = plot_connectivity(max_freq, ax=ax[0, idx], yticklabels=list(np.round(ei_ratio, decimals=2)),
                             xticklabels=list(np.round(io_ratio, decimals=2)), cmap=cm1)
    cax1.set_xlabel('intra/inter')
    cax1.set_ylabel('exc/inh')
    cax1.set_title(f'freq (J = {J})')
    cax2 = plot_connectivity(freq_pow, ax=ax[1, idx], yticklabels=list(np.round(ei_ratio, decimals=2)),
                             xticklabels=list(np.round(io_ratio, decimals=2)), cmap=cm2)
    cax2.set_xlabel('intra/inter')
    cax2.set_ylabel('exc/inh')
    cax2.set_title(f'amp (J = {J})')

plt.suptitle('EI-circuit sensitivity to population Coupling strengths (pcs)')
plt.tight_layout(pad=2.5, rect=(0.01, 0.01, 0.99, 0.96))
#fig.savefig("/home/rgast/Documents/Studium/PhD_Leipzig/Figures/BGTCS/eic_coupling", format="svg")
plt.show()
