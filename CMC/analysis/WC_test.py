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
T = 8000                                        # total simulation time in ms
delay = 1000
N = 15
m = 5
inp = np.zeros((int(T/dt), N), dtype='float32')
#inp[5000:15000, :] = 1.0 # external input to the population
dur = 50.0
ramp = 15.0
i = 0
while (i+1)*dur < T-delay:
    if i*dur > delay:
        i_tmp = i % m
        inp[int(((i+1)*dur+ramp)/dt):int(((i+2)*dur-ramp)/dt), i_tmp] = 1.0
    i += 1
inp = gaussian_filter1d(inp, sigma=0.5*ramp/dt, axis=0)
plt.plot(inp[:, 1])
plt.plot(inp[:, 2])
plt.show()

# circuit setup
# circuit = CircuitTemplate.from_yaml("../config/wc_templates/WC").apply()
C_ee = np.zeros((N, N))
C_ee[np.eye(N) == 0] = 2.0/N

C_ie = np.zeros((N, N))
C_ie[np.eye(N) == 0] = 2.5/N

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
result, t = compute_graph.run(T,
                              inputs={"all/E/E_op/I_ext": inp},
                              outputs={"meg": "all/E/E_op/meg"},
                              sampling_step_size=dts,
                              profile=True,
                              verbose=True,
                              )
# result, t = compute_graph.run(T,
#                               inputs={"E/E_op/I_ext": inp},
#                               outputs={"meg": "E/E_op/meg"},
#                               sampling_step_size=dts,
#                               profile=True,
#                               verbose=True,
#                               )

# visualization
result.mean(axis=1).plot()
plt.show()
