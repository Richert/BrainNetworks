# pyrates imports
from pyrates.frontend import EdgeTemplate, CircuitTemplate
from pyrates.backend import ComputeGraph
from pyrates.ir import CircuitIR

# additional imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

# general parameters
dt = 1e-1                                   # integration step size in s
dts = 1.0                                     # variable storage sub-sampling step size in s
sub = int(dts/dt)                              # sub-sampling rate
T = 8000                                        # total simulation time in ms
delay = 1000
reg_dur = 3500
dur = 50.0
ramp = 15.0

# network parameters
N = 100
p_e = 0.1
p_i = 0.1
k_e = 25.0
k_i = 10.0

# connectivity matrices
C_ee = np.ones((N, N))
C_ie = np.ones_like(C_ee)
n_e = int(np.ceil(p_e*N))
n_i = int(np.ceil(p_i*N))
for i in range(N):
    idx_e = np.random.randint(low=0, high=N, size=n_e)
    idx_i = np.random.randint(low=0, high=N, size=n_i)
    C_ee[i, idx_e] = k_e/n_e
    C_ie[i, idx_i] = k_i/n_i

# delay matrices
D_ee = np.zeros_like(C_ee)
indices_e = np.argwhere(C_ee > 0)
for idx in indices_e:
    D_ee[idx[0], idx[1]] = np.random.uniform(1.0, 5.0)
D_ie = np.zeros_like(C_ie)
indices_i = np.argwhere(C_ie > 0)
for idx in indices_i:
    D_ie[idx[0], idx[1]] = D_ee[idx[0], idx[1]] if idx in indices_e else np.random.uniform(1.0, 5.0)

ms = [5]
results = []

for m in ms:

    inp = np.zeros((int(T/dt), N), dtype='float32')

    i = 0
    inp_nodes = np.random.randint(0, N, m)
    while (i+1)*dur < T:
        if i*dur > delay:
            sequential = i*dur < delay+reg_dur
            i_tmp = i % m if sequential else int(np.random.uniform(0, m))
            inp[int(((i+1)*dur+ramp)/dt):int(((i+2)*dur-ramp)/dt), inp_nodes[i_tmp]] = 1.0
        i += 1
    inp = gaussian_filter1d(inp, sigma=0.5*ramp/dt, axis=0)

    # circuit setup
    circuit = CircuitIR()
    edge1 = EdgeTemplate.from_yaml("../config/wc_templates/EE_edge")
    edge2 = EdgeTemplate.from_yaml("../config/wc_templates/EI_edge")
    for idx in range(N):
        circuit.add_circuit(f'wc_{idx}', CircuitIR.from_yaml("../config/wc_templates/WC"))
    circuit.add_edges_from_matrix(source_var="E/E_op/m", target_var="E/E_op/I_e", template=edge1,
                                  nodes=[f'wc_{idx}' for idx in range(N)], weight=C_ee, delay=D_ee)
    circuit.add_edges_from_matrix(source_var="E/E_op/m", target_var="I/I_op/I_e", template=edge2,
                                  nodes=[f'wc_{idx}' for idx in range(N)], weight=C_ie, delay=D_ie)

    # circuit compilation and simulation
    compute_graph = circuit.compile(vectorization=True, backend='numpy', name='wc_net', step_size=dt, solver='euler')
    r, t = compute_graph.run(T,
                             #inputs={"all/E/E_op/I_ext": inp},
                             outputs={"meg": "all/E/E_op/meg"},
                             sampling_step_size=dts,
                             profile=True,
                             verbose=True,
                             )
    results.append(r)

# visualization
results[0].plot()
plt.figure()
for r in results:
    r.mean(axis=1).plot()
plt.legend([f"m = {m}" for m in ms])
plt.show()
plt.savefig(f'EEG_reg_rand_avg.svg', dpi=600, format='svg')
