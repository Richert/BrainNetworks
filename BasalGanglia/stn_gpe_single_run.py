from pyrates.utility import plot_timeseries, grid_search, plot_psd, plot_connectivity
from pyrates.ir import CircuitIR
import numpy as np
import matplotlib.pyplot as plt
from numba import jit


# parameters
dt = 1e-4
T = 2.0
dts = 1e-3

eic = CircuitIR.from_yaml("config/stn_gpe/net_qif_syn_adapt").compile(backend='numpy', step_size=dt, solver='scipy')
results, t = eic.run(simulation_time=T, sampling_step_size=dts, profile=True,
                     outputs={'r_e': 'stn/qif_stn/R_e',
                              'r_i': 'gpe/qif_gpe/R_i',
                              'v_e': 'stn/qif_stn/V_e',
                              'v_i': 'gpe/qif_gpe/V_i',
                              }
                     )

#print(results.iloc[-1, :])
results.plot()
plt.show()
