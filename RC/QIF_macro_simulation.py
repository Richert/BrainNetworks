from pyrates.utility.grid_search import grid_search
import numpy as np
from pyrates.utility.visualization import plot_connectivity
from matplotlib.pyplot import show

# Parameters
############

# general parameters
dt = 1e-4
dts = 1e-2
T = 130.
cutoff = 10.0

# model parameters
n_etas = 10
n_js = 10
etas = np.linspace(-1.0, 1.0, num=n_etas)
Js = np.linspace(2.0, 20.0, num=n_js)
params = {'eta': etas, 'J': Js}
param_map = {'eta': {'vars': ['Op_sfa_syns_noise/eta'], 'nodes': ['qif']},
             'Js': {'vars': ['Op_sfa_syns_noise/J'], 'nodes': ['qif']}
             }

# simulation
############

results, result_map = grid_search(circuit_template="qifs/QIF_sfa_syns_noise",
                                  param_grid=params,
                                  param_map=param_map,
                                  inputs={},
                                  outputs={"r": "qif/Op_sfa_syns_noise/r", "v": "qif/Op_sfa_syns_noise/v"},
                                  step_size=dt, simulation_time=T, permute_grid=True, sampling_step_size=dts,
                                  method='RK45')

# visualization
###############

sigmas = np.zeros((len(Js), len(etas)))
for i, J in enumerate(Js[::-1]):
    for j, eta in enumerate(etas):
        idx = result_map[result_map['eta'] == eta and result_map['J'] == J]
        r = results.loc[:, ('r', idx)]
        v = results.loc[:, ('v', idx)]
        w = np.complex(real=np.pi*r, imag=-v)
        sigmas[i, j] = (1-w)/(1+w)

plot_connectivity(sigmas, xticklabels=np.round(etas, decimals=2),
                  yticklabels=np.round(Js[::-1], decimals=2))
show()
