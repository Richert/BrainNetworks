from pyrates.utility import plot_connectivity, grid_search, create_cmap
import numpy as np
import matplotlib.pyplot as plt


# Parameters
############

# general parameters
dt = np.round(5e-4, decimals=5)
dts = 1e-2
T = 100.

# model parameters
n_samples = 5
r0 = (1.074832893*np.linspace(0.99, 1.01, n_samples)).tolist()
v0 = (-0.083029685988999999*np.linspace(0.99, 1.01, n_samples)).tolist()
e0 = (0.37855747705999998*np.linspace(0.94, 0.96, n_samples)).tolist()
a0 = (0.0044946096295*np.linspace(0.99, 1.01, n_samples)).tolist()
params = {'r0': r0, 'v0': v0, 'e0': e0, 'a0': a0}
param_map = {'r0': {'var': [('Op_e_adapt.0', 'r')],
                    'nodes': ['E.0']},
             'v0': {'var': [('Op_e_adapt.0', 'v')],
                    'nodes': ['E.0']},
             'e0': {'var': [('Op_e_adapt.0', 'e')],
                    'nodes': ['E.0']},
             'a0': {'var': [('Op_e_adapt.0', 'a')],
                    'nodes': ['E.0']}
             }

# simulation
############

results = grid_search(circuit_template="../config/cmc_templates.E_adapt",
                      param_grid=params,
                      param_map=param_map,
                      inputs={},
                      outputs={"r": ("E.0", "Op_e_adapt.0", "r"),
                               "v": ("E.0", "Op_e_adapt.0", "v"),
                               "e": ("E.0", "Op_e_adapt.0", "e")},
                      dt=dt, simulation_time=T, permute_grid=True, sampling_step_size=dts)

# visualization
###############

# color maps
#cm1 = create_cmap('pyrates_red', as_cmap=True, n_colors=16)
#cm2 = create_cmap('pyrates_green', as_cmap=True, n_colors=16)
#cm3 = create_cmap('pyrates_blue/pyrates_yellow', as_cmap=True, n_colors=16, pyrates_blue={'reverse': True},
#                  pyrates_yellow={'reverse': True})
#cm4 = create_cmap(palette_type='cubehelix', as_cmap=True, dark=0, light=1, n_colors=8)

fig, axes = plt.subplots(ncols=3, figsize=(15, 8))
for r in params['r0']:
    for v in params['v0']:
        for e in params['e0']:
            for a in params['a0']:
                data = results[r][v][e][a]
                axes[0].plot(data.loc[40.0:, 'v'], data.loc[40.0:, 'r'])
                axes[1].plot(data.loc[40.0:, 'e'], data.loc[40.0:, 'r'])
                axes[2].plot(data.loc[40.0:, 'e'], data.loc[40.0:, 'v'])

fig2, axes2 = plt.subplots(ncols=3, figsize=(15, 8))
for r in params['r0']:
    for v in params['v0']:
        for e in params['e0']:
            for a in params['a0']:
                data = results[r][v][e][a]
                axes2[0].plot(data.loc[:, 'v'], data.loc[:, 'r'])
                axes2[1].plot(data.loc[:, 'e'], data.loc[:, 'r'])
                axes2[2].plot(data.loc[:, 'e'], data.loc[:, 'v'])
plt.show()
