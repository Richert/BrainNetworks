from pyrates.ir import CircuitIR
import matplotlib.pyplot as plt
import os
import numpy as np

# parameters
dt = 5e-3
T = 100.0
dts = 1e-1

inp = np.zeros((int(T/dt), 1))
inp[2000:2100] = 1.0

syns = CircuitIR.from_yaml("config/stn_gpe/stn_syns_pop").compile(backend='numpy', step_size=dt, solver='euler')
results, t = syns.run(simulation_time=T, sampling_step_size=dts, profile=True,
                      outputs={'I_e': 'stn/stn_syns_op/I_ampa', 'I_i': 'stn/stn_syns_op/I_gabaa',
                               'G': 'stn/stn_syns_op/stn_fb'
                               },
                      inputs={
                          'stn/stn_syns_op/I_exc': inp,
                          'stn/stn_syns_op/I_inh': inp,
                              })
results.plot()
plt.show()
