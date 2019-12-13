from pyrates.utility import plot_timeseries, grid_search, plot_psd, plot_connectivity
from pyrates.ir import CircuitIR
import numpy as np
import matplotlib.pyplot as plt
from numba import jit


# parameters
dt = 5e-5
T = 2.0
dts = 1e-3

eic = CircuitIR.from_yaml("config/stn_gpe/net_qif_syn_adapt").compile(backend='numpy', step_size=dt, solver='euler')
results, t = eic.run(simulation_time=T, sampling_step_size=dts, profile=True,
                     outputs={'r_e': 'stn/qif_stn/R_e',
                              'r_i': 'gpe/qif_gpe/R_i',
                              #'v_e': 'stn/qif_stn/V_e',
                              #'v_i': 'gpe/qif_gpe/V_i',
                              }
                     )

#print(results.iloc[-1, :])
results.plot()

eic2 = CircuitIR.from_yaml("config/stn_gpe/net_stn_gpe").compile(backend='numpy', step_size=dt, solver='scipy')
results2, t = eic2.run(simulation_time=T, sampling_step_size=dts, profile=True,
                       outputs={'r_e': 'stn_gpe/qif_full/R_e',
                                'r_i': 'stn_gpe/qif_full/R_i',
                                #'v_e': 'stn_gpe/qif_full/V_e',
                                #'v_i': 'stn_gpe/qif_full/V_i',
                                }
                       )

#print(results.iloc[-1, :])
results2.plot()
plt.show()
