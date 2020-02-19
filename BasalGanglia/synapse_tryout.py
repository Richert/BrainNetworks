from pyrates.ir import CircuitIR
import matplotlib.pyplot as plt
import os
import numpy as np

# parameters
dt = 5e-3
T = 100.0
dts = 1e-1

inp = np.zeros((int(T/dt), 1))
inp[200:210] = 10.0

eic = CircuitIR.from_yaml("config/stn_gpe/synapse").compile(backend='numpy', step_size=dt, solver='euler')
results, t = eic.run(simulation_time=T, sampling_step_size=dts, profile=True, outputs={'I': 'syn/biexp/I'},
                     inputs={'syn/biexp/inp': inp})
results.plot()
plt.show()
