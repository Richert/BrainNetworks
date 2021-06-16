from pyrates.utility.grid_search import grid_search
import numpy as np
from pyrates.utility.visualization import Interactive2DParamPlot
import matplotlib.pyplot as plt
from pandas import DataFrame

# Parameters
############

# general parameters
dt = 1e-4
dts = 1e-2
T = 52.
cutoff = 2.0

# model parameters
n_etas = 10
n_js = 10
etas = np.linspace(-1.0, 1.0, num=n_etas)
Js = np.linspace(6.0, 15.0, num=n_js)
params = {'eta': etas, 'J': Js}
param_map = {'eta': {'vars': ['Op_sfa_syns_noise/eta'], 'nodes': ['qif']},
             'J': {'vars': ['Op_sfa_syns_noise/J'], 'nodes': ['qif']}
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

sigma_mean = np.zeros((len(Js), len(etas)))
sigma_range = np.zeros_like(sigma_mean)
sigmas = {}
for i, J in enumerate(Js):
    for j, eta in enumerate(etas):
        idx1 = result_map['eta'] == eta
        idx2 = result_map['J'] == J
        idx = result_map[idx1*idx2 == 1].index[0]
        r = results.loc[cutoff:, ('r', idx)]
        v = results.loc[cutoff:, ('v', idx)]
        w = np.conjugate(np.pi*r.values + 1j*v.values)
        sigma = np.real((1-w)/(1+w))
        sigma_mean[i, j] = np.mean(sigma)
        sigma_range[i, j] = np.max(sigma) - np.min(sigma)
        sigmas[idx] = sigma[:, 0].tolist()

df = DataFrame.from_dict(sigmas)
#df.index = results.index[results.index >= cutoff]

Interactive2DParamPlot(sigma_mean, df, x_values=etas, y_values=Js, param_map=result_map,
                       tmin=cutoff, x_key='eta', y_key='J', title='mean sigma', origin='lower')

Interactive2DParamPlot(sigma_range, df, x_values=etas, y_values=Js, param_map=result_map,
                       tmin=cutoff, x_key='eta', y_key='J', title='max(sigma) - min(sigma)', origin='lower')
plt.show()
