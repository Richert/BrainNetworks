from pyrates.frontend import CircuitTemplate
from pyrates.utility.grid_search import grid_search
from pyrates.utility.visualization import plot_connectivity
import matplotlib.pyplot as plt
import os
import numpy as np

# parameters
dt = 5e-4
T = 50.0
start = int(10.0/dt)
stop = int(12.0/dt)
dts = 1e-2

inp = np.zeros((int(T/dt), 1))
inp[start:stop] = 1.0

# target: delayed biexponential feedback
biexp = CircuitTemplate.from_yaml("config/stn_gpe/biexp_gamma")
r1 = biexp.run(simulation_time=T, sampling_step_size=dts, inputs={'n1/biexp_rate/I_ext': inp},
               outputs={'r': 'n1/biexp_rate/r'}, backend='numpy', step_size=dt, solver='euler')
# fig, ax = plt.subplots()
# ax.plot(r1['r'])
# plt.show()

# approximation: gamma-distributed feedback
param_grid = {'d': np.asarray([5.0, 6.0, 7.0]),
              's': np.asarray([1.0, 1.5, 2.0])}
param_map = {'d': {'vars': ['delay'], 'edges': [('n1/biexp_rate/r2', 'n1/biexp_rate/r_in')]},
             's': {'vars': ['spread'], 'edges': [('n1/biexp_rate/r2', 'n1/biexp_rate/r_in')]}}

out_var = 'n1/biexp_rate/r'
r2, r_map = grid_search("config/stn_gpe/biexp_gamma", param_grid, param_map, step_size=dt, simulation_time=T,
                        sampling_step_size=dts, permute_grid=True, backend='numpy', solver='euler',
                        outputs={'r': out_var}, inputs={'n1/biexp_rate/I_ext': inp}, clear=False)

# calculate difference between target and approximation
n = len(param_grid['d'])
m = len(param_grid['s'])
alpha = 0.95
error = np.zeros((n, m))
indices = [['_'for j in range(m)] for i in range(n)]
for idx in r_map.index:
    idx_r = np.argmin(np.abs(param_grid['d'] - r_map.at[idx, 'd']))
    idx_c = np.argmin(np.abs(param_grid['s'] - r_map.at[idx, 's']))
    r = r2.loc[:, ('r', f"{idx}/{out_var}")]
    diff = r - r1.loc[:, 'r']
    d, s = r_map.loc[idx, 'd'], r_map.loc[idx, 's']
    order = int(np.round((d/s)**2))
    error[idx_r, idx_c] = alpha*np.sqrt(diff.T @ diff) + (1-alpha)*order
    print(f"delay = {d}, spread = {s}, order = {order}, rate = {order/d}, error = {error[idx_r, idx_c]}")
    indices[idx_r.squeeze()][idx_c.squeeze()] = idx

# display error
fig, ax = plt.subplots()
ax = plot_connectivity(error, xticklabels=param_grid['s'], yticklabels=param_grid['d'], ax=ax)
ax.set_xlabel('s')
ax.set_ylabel('d')
plt.tight_layout()

# display winner together with target
fig2, ax2 = plt.subplots()
winner = np.argmin(error)
idx = np.asarray(indices).flatten()[winner]
ax2.plot(r1.loc[:, 'r'])
ax2.plot(r2.loc[:, ('r', f"{idx}/{out_var}")])
plt.legend(['discrete', 'gamma'])
ax2.set_title(f"delay = {r_map.loc[idx, 'd']}, spread = {r_map.loc[idx, 's']}, error = {error.flatten()[winner]}")
plt.tight_layout()

plt.show()
