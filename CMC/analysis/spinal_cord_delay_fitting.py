from pyrates.frontend import CircuitTemplate
from pyrates.utility.grid_search import grid_search
from pyrates.utility.visualization import plot_connectivity
import matplotlib.pyplot as plt
import numpy as np

# parameter definition
dt = 1e-3
dts = 1e-2
cutoff = 100.0
T = 200.0 + cutoff
start = int((0 + cutoff)/dt)
dur = int(5/(0.6*dt))
steps = int(T/dt)
inp = np.zeros((steps, 1))
inp[start:start+dur] = 0.6

# target: delayed biexponential response of the alpha or renshaw neuron
path = "../config/spinal_cord/sc"
neuron = 'alpha'
target_var = 'I_ampa'
model = CircuitTemplate.from_yaml(path).apply().compile(backend='numpy', step_size=dt, solver='euler')
r1 = model.run(simulation_time=T, sampling_step_size=dts, inputs={'m1/m1_dummy/m_in': inp},
               outputs={neuron: f'{neuron}/{neuron}_op/{target_var}'})
model.clear()
r1.plot()
plt.show()

# approximation: gamma-distributed feedback
source = 'm1'
param_grid = {'d': np.asarray([1.5, 2.0, 2.5]),
              's': np.asarray([0.4, 0.6, 0.8, 1.0])}
param_map = {'d': {'vars': ['delay'], 'edges': [(source, neuron)]},
             's': {'vars': ['spread'], 'edges': [(source, neuron)]}}

r2, r_map = grid_search(path, param_grid, param_map, step_size=dt, simulation_time=T,
                        sampling_step_size=dts, permute_grid=True,
                        init_kwargs={'backend': 'numpy', 'step_size': dt, 'solver': 'scipy'},
                        outputs={neuron: f'{neuron}/{neuron}_op/{target_var}'},
                        inputs={'m1/m1_dummy/m_in': inp})

# calculate difference between target and approximation
n = len(param_grid['d'])
m = len(param_grid['s'])
alpha = 0.95   # controls trade-off between accuracy and complexity of gamma-kernel convolution. alpha = 1.0 for max accuracy.
error = np.zeros((n, m))
indices = [['_'for j in range(m)] for i in range(n)]
for idx in r_map.index:
    idx_r = np.argmin(np.abs(param_grid['d'] - r_map.at[idx, 'd']))
    idx_c = np.argmin(np.abs(param_grid['s'] - r_map.at[idx, 's']))
    r = r2.loc[:, (neuron, idx)]
    diff = r - r1.loc[:, neuron]
    d, s = r_map.loc[idx, 'd'], r_map.loc[idx, 's']
    order = (d/s)**2
    error[idx_r, idx_c] = alpha*np.sqrt(diff.T @ diff).iloc[0, 0] + (1-alpha)*order
    print(f"delay = {d}, spread = {s}, order = {order}, rate = {order/d}")
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
ax2.plot(r1.loc[:, neuron])
ax2.plot(r2.loc[:, (neuron, idx)])
ax2.set_title(f"delay = {r_map.loc[idx, 'd']}, spread = {r_map.loc[idx, 's']}")
plt.tight_layout()

plt.show()
