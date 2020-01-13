from pyrates.utility import grid_search
import matplotlib.pyplot as plt

# define system parameters
param_map = {
        'k_ee': {'vars': ['qif_stn/k_ee'], 'nodes': ['stn']},
        'k_ei': {'vars': ['qif_stn/k_ei'], 'nodes': ['stn']},
        'k_ie': {'vars': ['qif_gpe/k_ie'], 'nodes': ['gpe']},
        'k_ii': {'vars': ['qif_gpe/k_ii'], 'nodes': ['gpe']},
        'eta_e': {'vars': ['qif_stn/eta_e'], 'nodes': ['stn']},
        'eta_i': {'vars': ['qif_gpe/eta_i'], 'nodes': ['gpe']},
        'eta_str': {'vars': ['qif_gpe/eta_str'], 'nodes': ['gpe']},
        'eta_tha': {'vars': ['qif_gpe/eta_tha'], 'nodes': ['gpe']},
        'alpha': {'vars': ['qif_stn/alpha', 'qif_gpe/alpha'], 'nodes': ['stn', 'gpe']},
        'delta_e': {'vars': ['qif_stn/delta'], 'nodes': ['stn']},
        'delta_i': {'vars': ['qif_stn/delta'], 'nodes': ['gpe']},
        'd': {'vars': ['delay'], 'edges': [('stn', 'gpe'), ('gpe', 'stn')]},
        's': {'vars': ['spread'], 'edges': [('stn', 'gpe'), ('gpe', 'stn')]}
    }

param_grid = {
        'd': [0.0, 0.004],
        's': [0.0, 0.002]
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
               'delta_e', 'delta_i', 'd', 's']
for c_dict in conditions:

    for key in models_vars:
        param_grid[key].append(param_grid[key][0] if key not in c_dict else c_dict[key])

for key in param_grid.copy():
    if key not in models_vars:
        param_grid.pop(key)

# define simulation parameters
dt = 5e-6
T = 2000.0
dts = 1.0

# perform simulation
results, _ = grid_search(circuit_template="config/stn_gpe/net_qif_syn_adapt",
                         param_grid=param_grid,
                         param_map=param_map,
                         simulation_time=T,
                         step_size=dt,
                         sampling_step_size=dts,
                         permute_grid=True,
                         inputs={},
                         outputs={},
                         init_kwargs={'backend': 'numpy', 'solver': 'scipy', 'step_size': dt},
                         )
results.plot()
plt.show()
