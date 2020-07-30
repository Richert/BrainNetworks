from pyrates.frontend import CircuitTemplate
from pyrates.utility import grid_search
import matplotlib.pyplot as plt
import os
import numpy as np

# parameters
dt = 5e-3
T = 100.0
dts = 1e-1

inp = np.zeros((int(T/dt), 1))
inp[2000:2100] = 1.0

# target: delayed biexponential feedback
biexp = CircuitTemplate.from_yaml("config/stn_gpe/biexp").apply(node_values={'n/biexp_simple/tau_r': 2.0,
                                                                             'n/biexp_simple/alpha': 0.05*10.0}
                                                                ).compile(backend='numpy', step_size=dt, solver='euler')
r1 = biexp.run(simulation_time=T, sampling_step_size=dts, inputs={'n/biexp_simple/r_in': inp},
               outputs={'I': 'n/biexp_simple/I'})
biexp.clear()
r1.plot()
plt.show()

# # approximation: gamma-distributed feedback
# param_grid = {'d': [2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5], 's': [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]}
# param_map = {'d': {'vars': ['delay'], 'edges': [('n1', 'n1')]}, 's': {'vars': ['spread'], 'edges': [('n1', 'n1')]}}
#
# r2, r_map = grid_search("config/stn_gpe/biexp_gamma", param_grid, param_map, step_size=dt, simulation_time=T,
#                         sampling_step_size=dts, permute_grid=True,
#                         init_kwargs={'backend': 'numpy', 'step_size': dt, 'solver': 'euler'},
#                         outputs={'r': 'n1/biexp_rate/r'}, inputs={'n1/biexp_rate/I_ext': inp})
#
# # calculate difference between target and approximation
# n = len(param_grid['d'])
# m = len(param_grid['s'])
# error = np.zeros((n, m))
# indices = [['_'for j in range(m)] for i in range(n)]
# for idx in r_map.index:
#     idx_r = np.argwhere(param_grid['d'] == r_map.loc[idx, 'd'])
#     idx_c = np.argwhere(param_grid['s'] == r_map.loc[idx, 's'])
#     r = r2.loc[:, ('r', idx)]
#     diff = r - r1.loc[:, 'r']
#     error[idx_r, idx_c] = np.sqrt(diff.T @ diff)
#     indices[idx_r.squeeze()][idx_c.squeeze()] = idx
#
# # display error
# fig, ax = plt.subplots()
# plot_connectivity(error, xticklabels=param_grid['s'], yticklabels=param_grid['d'], ax=ax)
# ax.set_xlabel('s')
# ax.set_ylabel('d')
# plt.tight_layout()
#
# # display winner together with target
# fig2, ax2 = plt.subplots()
# winner = np.argmin(error)
# idx = np.asarray(indices).flatten()[winner]
# ax2.plot(r1.loc[:, 'r'])
# ax2.plot(r2.loc[:, ('r', idx)])
# ax2.set_title(f"delay = {r_map.loc[idx, 'd']}, spread = {r_map.loc[idx, 's']}")
# plt.tight_layout()
#
# plt.show()
