from pyrates.utility import grid_search
import matplotlib.pyplot as plt

# define system parameters
param_map = {
        'k_ee': {'vars': ['qif_full/k_ee'], 'nodes': ['stn_gpe']},
        'k_ei': {'vars': ['qif_full/k_ei'], 'nodes': ['stn_gpe']},
        'k_ie': {'vars': ['qif_full/k_ie'], 'nodes': ['stn_gpe']},
        'k_ii': {'vars': ['qif_full/k_ii'], 'nodes': ['stn_gpe']},
        'eta_e': {'vars': ['qif_full/eta_e'], 'nodes': ['stn_gpe']},
        'eta_i': {'vars': ['qif_full/eta_i'], 'nodes': ['stn_gpe']},
        'eta_str': {'vars': ['qif_full/eta_str'], 'nodes': ['stn_gpe']},
        'eta_tha': {'vars': ['qif_full/eta_tha'], 'nodes': ['stn_gpe']},
        'alpha': {'vars': ['qif_full/alpha'], 'nodes': ['stn_gpe']},
        'delta_e': {'vars': ['qif_full/delta_e'], 'nodes': ['stn_gpe']},
        'delta_i': {'vars': ['qif_full/delta_i'], 'nodes': ['stn_gpe']},
        'd': {'vars': ['qif_full/d_e', 'qif_full/d_i'], 'nodes': ['stn_gpe']}
    }

param_grid = {
        'k_ee': [2.55],
        'k_ei': [79.77],
        'k_ie': [50.09],
        'k_ii': [1.13],
        'eta_e': [-4.93],
        'eta_i': [12.92],
        'eta_str': [-2.24],
        'eta_tha': [2.88],
        'alpha': [2.35],
        'k_ee_pd': [10.98],
        'k_ei_pd': [8.17],
        'k_ie_pd': [21.28],
        'k_ii_pd': [19.73],
        'eta_e_pd': [-5.99],
        'eta_i_pd': [-0.39],
        'eta_str_pd': [-2.94],
        'eta_tha_pd': [5.36],
        'delta_e': [2.13],
        'delta_i': [2.88],
        'delta_e_pd': [-1.54],
        'delta_i_pd': [-1.72],
        'd': [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008],
        's': [0.0, 0.0001, 0.0002, 0.0004, 0.0008, 0.0016, 0.0032]
    }

# define simulation conditions
conditions = [
    # {'k_ie': 0.0},  # STN blockade
    # {'k_ii': 0.0, 'eta_str': 0.0},  # GABAA blockade in GPe
    # {'k_ie': 0.0, 'k_ii': 0.0, 'eta_str': 0.0},  # STN blockade and GABAA blockade in GPe
    # {'k_ie': 0.0, 'eta_tha': 0.0},  # AMPA + NMDA blocker in GPe
    # {'k_ei': 0.0},  # GABAA antagonist in STN
    # {'k_ei': param_grid['k_ei'][0] + param_grid['k_ei_pd'][0],
    #  'k_ie': param_grid['k_ie'][0] + param_grid['k_ie_pd'][0],
    #  'k_ee': param_grid['k_ee'][0] + param_grid['k_ee_pd'][0],
    #  'k_ii': param_grid['k_ii'][0] + param_grid['k_ii_pd'][0],
    #  'eta_e': param_grid['eta_e'][0] + param_grid['eta_e_pd'][0],
    #  'eta_i': param_grid['eta_i'][0] + param_grid['eta_i_pd'][0],
    #  'eta_str': param_grid['eta_str'][0] + param_grid['eta_str_pd'][0],
    #  'eta_tha': param_grid['eta_tha'][0] + param_grid['eta_tha_pd'][0],
    #  'delta_e': param_grid['delta_e'][0] + param_grid['delta_e_pd'][0],
    #  'delta_i': param_grid['delta_i'][0] + param_grid['delta_i_pd'][0],
    #  }  # parkinsonian condition
              ]

models_vars = ['k_ie', 'k_ii', 'k_ei', 'k_ee', 'eta_e', 'eta_i', 'eta_str', 'eta_tha', 'alpha',
               'delta_e', 'delta_i']
for c_dict in conditions:

    for key in models_vars:
        param_grid[key].append(param_grid[key][0] if key not in c_dict else c_dict[key])

for key in param_grid.copy():
    if key not in models_vars:
        param_grid.pop(key)

# define simulation parameters
dt = 5e-6
T = 2000.0
dts = 1e-1

# perform simulation
results, _ = grid_search(circuit_template="config/stn_gpe/net_stn_gpe",
                         param_grid=param_grid,
                         param_map=param_map,
                         simulation_time=T,
                         dt=dt,
                         sampling_step_size=dts,
                         permute_grid=False,
                         inputs={},
                         outputs={},
                         init_kwargs={'backend': 'numpy', 'solver': 'scipy', 'step_size': dt},
                         )
results.plot()
for col in results.columns.values:
    print(col, results.loc[:, col].iloc[-1])
plt.show()
