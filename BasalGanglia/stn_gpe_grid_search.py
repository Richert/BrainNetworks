import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyrates.utility.grid_search import grid_search

param_map = {
    'k_ee': {'vars': ['qif_stn/k_ee'],
             'nodes': ['stn']},
    'k_ei': {'vars': ['qif_stn/k_ei'],
             'nodes': ['stn']},
    'k_ie': {'vars': ['qif_gpe/k_ie'],
             'nodes': ['gpe']},
    'k_ii': {'vars': ['qif_gpe/k_ii'],
             'nodes': ['gpe']},
    'eta_e': {'vars': ['qif_stn/eta_e'],
              'nodes': ['stn']},
    'eta_i': {'vars': ['qif_gpe/eta_i'],
              'nodes': ['gpe']},
    'alpha': {'vars': ['qif_gpe/alpha'],
              'nodes': ['gpe']}
}

param_grid = {'eta_e': np.linspace(-10.0, 10.0, 10), 'eta_i': np.linspace(-10.0, 10.0, 10)}

T = 5.
dt = 1e-4
dts = 1e-3

results, result_map = grid_search(
    circuit_template="config/stn_gpe/net_qif_syn_adapt",
    param_grid=param_grid,
    param_map=param_map,
    simulation_time=T,
    dt=dt,
    permute=True,
    sampling_step_size=dts,
    inputs={},
    outputs={'r_e': "stn/qif_stn/R_e",
             'r_i': 'gpe/qif_gpe/R_i'},
    init_kwargs={
        'backend': 'numpy', 'solver': 'scipy', 'step_size': dt}
)

results.plot()
print(results.head(3))
plt.show()
