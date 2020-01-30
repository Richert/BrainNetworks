from pyrates.utility import grid_search
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np
import scipy.io as scio

# define system parameters
param_map = {'k': {'vars': ['qif_full/k'], 'nodes': ['stn_gpe']},
             'eta_str': {'vars': ['qif_full/eta_str'], 'nodes': ['stn_gpe']}}

param_grid = {
        'k': [1.6, 1.8, 2.0, 2.2],
        'eta_str': [-50.0, -100.0],
    }

# define simulation parameters
dt = 1e-2
T = 21000.0
dts = 1e-1

# perform simulation
results, result_map = grid_search(circuit_template="config/stn_gpe/net_stn_gpe",
                                  param_grid=param_grid,
                                  param_map=param_map,
                                  simulation_time=T,
                                  step_size=dt,
                                  sampling_step_size=dts,
                                  permute_grid=True,
                                  inputs={},
                                  outputs={'Ie': 'stn_gpe/qif_full/I_ee',
                                           'Ii': 'stn_gpe/qif_full/I_ei'},
                                  init_kwargs={'backend': 'numpy', 'solver': 'scipy', 'step_size': dt},
                                  )
results = results * 1e3

#fig = plt.figure(figsize=(12, 10), tight_layout=True)
#grid = gs.GridSpec(6, 2)
#results.plot()
# codim 1
# for i, k in enumerate(param_grid['k']):
#     r, c = i % 6, i // 6
#     ax_tmp = fig.add_subplot(grid[r, c])
#     idx = result_map.index[result_map.loc[:, 'k'] == k]
#     for c, idx_tmp in zip(['r', 'g', 'b'], idx):
#         ax_tmp.plot(results.loc[1800:, ('R', idx_tmp)], c=c)
#     ax_tmp.set_title(f"k = {k}")
#
# plt.savefig(f"test_fig.svg", format='svg', dpi=600)
#plt.show()

results_dict = {}
for key in result_map.index:
    data1, data2 = results.loc[:, ('Ie', key)].values, results.loc[:, ('Ii', key)].values
    results_dict[key] = {"k": result_map.loc[key, 'k'], 'eta_str': result_map.loc[key, 'eta_str'],
                         "data": data1 + data2}
scio.savemat('PAC_timeseries.mat', mdict=results_dict, long_field_names=True)
