from pyrates.utility.grid_search import grid_search
import numpy as np


# Parameters
############

# general parameters
dt = 1e-4
dts = 1e-2
T = 130.
cutoff = 10.0

# model parameters
n_etas = 20
n_alphas = 5
etas = np.linspace(-6.5, -4.5, num=n_etas)
alphas = np.linspace(0.0, 0.1, num=n_alphas)
params = {'eta': etas, 'alpha': alphas}
param_map = {'eta': {'vars': ['Op_sd_exp/eta'], 'nodes': ['qif']},
             'alpha': {'vars': ['weight'], 'edges': [('inp', 'qif')]}
             }

# simulation of stuart-landau-driven system
###########################################

results_sl, result_map_sl = grid_search(circuit_template="qifs/QIF_sd_sl",
                                        param_grid=params,
                                        param_map=param_map,
                                        inputs={},
                                        outputs={"r": "qif/Op_sd_exp/r", "v": "qif/Op_sd_exp/v"},
                                        step_size=dt, simulation_time=T, permute_grid=True, sampling_step_size=dts,
                                        method='RK45')

# simulation of lorenz-driven system
####################################

results_lo, result_map_lo = grid_search(circuit_template="qifs/QIF_sd_lorenz",
                                        param_grid=params,
                                        param_map=param_map,
                                        inputs={},
                                        outputs={"r": "qif/Op_sd_exp/r", "v": "qif/Op_sd_exp/v"},
                                        step_size=dt, simulation_time=T, permute_grid=True, sampling_step_size=dts,
                                        method='RK45')

# store results
###############

results_sl.to_pickle("results/stuart_landau_ts.pkl")
result_map_sl.to_pickle("results/stuart_landau_map.pkl")
results_lo.to_pickle("results/lorenz_ts.pkl")
result_map_lo.to_pickle("results/lorenz_map.pkl")
