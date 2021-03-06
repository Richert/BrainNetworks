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

# population numbers
N = 250
N_c = 50
N_b = N - N_c

# connection probabilities
p_ee_cc = 0.1
p_ee_bb = 0.1
p_ee_bc = 0.05
p_ie_cc = 0.2
p_ie_bb = 0.2

# connection strengths
k_e = 10.0
k_i = 1.0

# connectivity matrices
C_ee_cc = np.random.randn(N_c, N_c)
C_ee_cc[np.eye(N_c, dtype=np.int32)] = 0
c_sorted = np.sort(C_ee_cc.flatten())
threshold = c_sorted[int(p_ee_cc*len(c_sorted))]
C_ee_cc[C_ee_cc < threshold] = 0.0
C_ee_cc[C_ee_cc >= threshold] = 1.0

C_ie_cc = np.random.randn(N_c, N_c)
C_ie_cc[np.eye(N_c, dtype=np.int32)] = 0
c_sorted = np.sort(C_ie_cc.flatten())
threshold = c_sorted[int(p_ie_cc*len(c_sorted))]
C_ie_cc[C_ie_cc < threshold] = 0.0
C_ie_cc[C_ie_cc >= threshold] = 1.0

C_ee_bb = np.random.randn(N_b, N_b)
C_ee_bb[np.eye(N_b, dtype=np.int32)] = 0
c_sorted = np.sort(C_ee_bb.flatten())
threshold = c_sorted[int(p_ee_bb*len(c_sorted))]
C_ee_bb[C_ee_bb < threshold] = 0.0
C_ee_bb[C_ee_bb >= threshold] = 1.0

C_ie_bb = np.random.randn(N_b, N_b)
C_ie_bb[np.eye(N_b, dtype=np.int32)] = 0
c_sorted = np.sort(C_ie_bb.flatten())
threshold = c_sorted[int(p_ie_bb*len(c_sorted))]
C_ie_bb[C_ie_bb < threshold] = 0.0
C_ie_bb[C_ie_bb >= threshold] = 1.0

C_ee_bc = np.random.randn(N, N)
C_ee_bc[0:N_c, 0:N_c] = 0.0
C_ee_bc[N_c:, N_c:] = 0.0
C_ee_bc[np.eye(N, dtype=np.int32)] = 0
c_sorted = np.sort(C_ee_bc.flatten())
threshold = c_sorted[int(p_ee_bc*len(c_sorted))]
C_ee_bc[C_ee_bc < threshold] = 0.0
C_ee_bc[C_ee_bc >= threshold] = 1.0
C_ee_bc[0:N_c, 0:N_c] = C_ee_cc
C_ee_bc[N_c:, N_c:] = C_ee_bb

C_ie_bc = np.zeros((N, N))
C_ie_bc[0:N_c, 0:N_c] = C_ie_cc
C_ie_bc[N_c:, N_c:] = C_ie_bb

for i in range(N):
    for C, c_scale in zip([C_ee_bc, C_ie_bc], [k_e, k_i]):
        c_max = np.max(C[i, :])
        if c_max > 0:
            C[i, :] /= np.sum(C[i, :])
            C[i, :] *= c_scale

ms = [5]
results = []

for m in ms:

    inp = np.zeros((int(T/dt), N), dtype='float32')

    i = 0
    inp_nodes = np.random.randint(0, N_c, m)
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
                                  nodes=[f'wc_{idx}' for idx in range(N)], weight=C_ee_bc)
    circuit.add_edges_from_matrix(source_var="E/E_op/m", target_var="I/I_op/I_e", template=edge2,
                                  nodes=[f'wc_{idx}' for idx in range(N)], weight=C_ie_bc)

    # circuit compilation and simulation
    compute_graph = circuit.compile(vectorization=True, backend='numpy', name='wc_net', step_size=dt, solver='euler')
    r, t = compute_graph.run(T,
                             inputs={"all/E/E_op/I_ext": inp},
                             outputs={"meg": "all/E/E_op/meg"},
                             sampling_step_size=dts,
                             profile=True,
                             verbose=True,
                             )
    results.append(r)

# visualization
plt.figure()
for r in results:
    r.mean(axis=1).plot()
plt.legend([f"m = {m}" for m in ms])
plt.show()
plt.savefig(f'EEG_reg_rand_avg.svg', dpi=600, format='svg')
