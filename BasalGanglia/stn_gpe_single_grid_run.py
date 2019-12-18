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
        'eta_str': {'vars': ['qif_gpe/eta_i'], 'nodes': ['gpe']},
        'eta_tha': {'vars': ['qif_gpe/eta_i'], 'nodes': ['gpe']},
        'alpha': {'vars': ['qif_gpe/alpha'], 'nodes': ['gpe']},
        'k_ee_pd': {'vars': ['qif_stn/k_ee'], 'nodes': ['stn']},
        'k_ei_pd': {'vars': ['qif_stn/k_ei'], 'nodes': ['stn']},
        'k_ie_pd': {'vars': ['qif_gpe/k_ie'], 'nodes': ['gpe']},
        'k_ii_pd': {'vars': ['qif_gpe/k_ii'], 'nodes': ['gpe']},
        'eta_e_pd': {'vars': ['qif_stn/eta_e'], 'nodes': ['stn']},
        'eta_i_pd': {'vars': ['qif_gpe/eta_i'], 'nodes': ['gpe']},
        'eta_str_pd': {'vars': ['qif_gpe/eta_i'], 'nodes': ['gpe']},
        'eta_tha_pd': {'vars': ['qif_gpe/eta_i'], 'nodes': ['gpe']},
        'alpha_pd': {'vars': ['qif_gpe/alpha'], 'nodes': ['gpe']},
        'delta_e': {'vars': ['qif_stn/delta'], 'nodes': ['stn']},
        'delta_i': {'vars': ['qif_gpe/delta'], 'nodes': ['gpe']},
        'delta_e_pd': {'vars': ['qif_stn/delta'], 'nodes': ['stn']},
        'delta_i_pd': {'vars': ['qif_gpe/delta'], 'nodes': ['gpe']}
    }

param_grid = {
        'k_ee': [3.01],
        'k_ei': [55.34],
        'k_ie': [32.06],
        'k_ii': [28.69],
        'eta_e': [-6.07],
        'eta_i': [4.87],
        'eta_str': [-3.64],
        'eta_tha': [8.45],
        'alpha': [5.0],
        'k_ee_pd': [10.0],
        'k_ei_pd': [50.0],
        'k_ie_pd': [50.0],
        'k_ii_pd': [25.0],
        'eta_e_pd': [0.0],
        'eta_i_pd': [0.0],
        'eta_str_pd': [-10.0],
        'eta_tha_pd': [0.0],
        'delta_e': [1.55],
        'delta_i': [1.55],
        'delta_e_pd': [-1.0],
        'delta_i_pd': [-1.0],
    }

# define simulation conditions
conditions = [{'k_ie': 0.0},  # STN blockade
              {'k_ii': 0.0, 'eta_str': 0.0},  # GABAA blockade in GPe
              {'k_ie': 0.0, 'k_ii': 0.0, 'eta_str': 0.0},  # STN blockade and GABAA blockade in GPe
              {'k_ie': 0.0, 'eta_tha': 0.0},  # AMPA + NMDA blocker in GPe
              {'k_ei': 0.0},  # GABAA antagonist in STN
              {'k_ei': param_grid['k_ei'][0] + param_grid['k_ei_pd'][0],
               'k_ie': param_grid['k_ie'][0] + param_grid['k_ie_pd'][0],
               'k_ee': param_grid['k_ee'][0] + param_grid['k_ee_pd'][0],
               'k_ii': param_grid['k_ii'][0] + param_grid['k_ii_pd'][0],
               'eta_e': param_grid['eta_e'][0] + param_grid['eta_e_pd'][0],
               'eta_i': param_grid['eta_i'][0] + param_grid['eta_i_pd'][0],
               'eta_str': param_grid['eta_str'][0] + param_grid['eta_str_pd'][0],
               'eta_tha': param_grid['eta_tha'][0] + param_grid['eta_tha_pd'][0],
               'delta_e': param_grid['delta_e'][0] + param_grid['delta_e_pd'][0],
               'delta_i': param_grid['delta_i'][0] + param_grid['delta_i_pd'][0],
               }  # parkinsonian condition
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
dt = 5e-5
T = 2.0
dts = 1e-3

# perform simulation
results, _ = grid_search(circuit_template="config/stn_gpe/net_qif_syn_adapt",
                         param_grid=param_grid,
                         param_map=param_map,
                         simulation_time=T,
                         dt=dt,
                         sampling_step_size=dts,
                         permute_grid=False,
                         inputs={},
                         outputs={'r_e': 'stn/qif_stn/R_e', 'r_i': 'gpe/qif_gpe/R_i'},
                         init_kwargs={'backend': 'numpy', 'solver': 'scipy', 'step_size': dt},
                         )
results.plot()
plt.show()
