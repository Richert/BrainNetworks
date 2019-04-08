from pyrates.utility import plot_timeseries, plot_psd, plot_connectivity, create_cmap, adapt_circuit
from pyrates.frontend import CircuitTemplate
from pyrates.backend import ComputeGraph
from pyrates.ir import CircuitIR
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from copy import deepcopy

# general parameters
dt = 1e-5
T = 4.
N = 100

# input parameters
mu = 3.
input_start = int(1./dt)
input_end = int(3./dt)
input_constant = np.zeros((int(T/dt), N))
input_constant[input_start:input_end, :] = mu

# network connectivity
C = np.ones((N, N))
conns = DataFrame(C, columns=[f'qif_{idx}/E.0/Op_spike.0/s' for idx in range(N)])
conns.index = [f'qif_{idx}/E.0/Op_e_v_base.0/s' for idx in range(N)]

# firing thresholds
etas = np.zeros((N,))
for i in range(N):
    etas[i] = -5 + np.random.randn() * 1.0

# set up template
template = CircuitTemplate.from_yaml("../config/qif_templates.E_base")

# set up intermediate representation
circuits = {}
for idx in range(N):
    circuit = adapt_circuit(deepcopy(template).apply(),
                            params={'eta': etas[idx]},
                            param_map={'eta': {'var': [('Op_e_v_base.0', 'eta')],
                                               'nodes': ['E.0']}})

    circuits[f'qif_{idx}'] = circuit
circuit = CircuitIR.from_circuits(label='net', circuits=circuits, connectivity=conns)

# set up compute graph
net = ComputeGraph(circuit, dt=dt, vectorization='full')

# run simulations
results = net.run(T, inputs={('E', 'Op_e_v_base.0', 'i_in'): input_constant},
                  outputs={'s': ('E', 'Op_spike.0', 's')},
                  verbose=True, sampling_step_size=1e-3)

# plotting
fig, axes = plt.subplots(ncols=2, figsize=(15, 5))
axes[0].matshow(results.values.T, aspect=100., cmap='Greys')
axes[1].plot(np.mean(results.values, axis=1))
plt.show()
