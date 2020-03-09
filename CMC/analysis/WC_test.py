# pyrates imports
from pyrates.frontend import EdgeTemplate, CircuitTemplate
from pyrates.backend import ComputeGraph
from pyrates.ir import CircuitIR

# additional imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

# parameters
dt = 1e-1                                   # integration step size in s
dts = 1.0                                     # variable storage sub-sampling step size in s
sub = int(dts/dt)                              # sub-sampling rate
T = 6000                                        # total simulation time in ms
delay = 1000
reg_dur = 2000
N = 100
dur = 50.0
ramp = 15.0

C_ee = np.random.randn(N, N)
C_ee[np.eye(N, dtype=np.int32)] = 0
c_sorted = np.sort(C_ee.flatten())
threshold = c_sorted[int(0.9*len(c_sorted))]
C_ee[C_ee < threshold] = 0.0

C_ie = np.random.randn(N, N)
C_ie[np.eye(N, dtype=np.int32)] = 0
c_sorted = np.sort(C_ie.flatten())
threshold = c_sorted[int(0.7*len(c_sorted))]
C_ee[C_ie < threshold] = 0.0

for i in range(N):
    for C, c_scale in zip([C_ee, C_ie], [2.0, 0.5]):
        c_max = np.max(C[i, :])
        if c_max > 0:
            C[i, :] /= np.sum(C[i, :])
            C[i, :] *= c_scale

ms = [5, 10, 15]
results = []

for m in ms:

    inp = np.zeros((int(T/dt), N), dtype='float32')

    i = 0
    inp_nodes = np.random.randint(0, N, m)
    while (i+1)*dur < T-delay:
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
                                  nodes=[f'wc_{idx}' for idx in range(N)], weight=C_ee)
    circuit.add_edges_from_matrix(source_var="E/E_op/m", target_var="I/I_op/I_e", template=edge2,
                                  nodes=[f'wc_{idx}' for idx in range(N)], weight=C_ie)

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
