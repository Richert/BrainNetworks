from pyrates.utility import plot_timeseries, grid_search, plot_psd, plot_connectivity
from pyrates.ir import CircuitIR
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

# parameters
dt = 1e-2
T = 2000.0
dts = 1.0
ctx = np.zeros((int(T/dt), 1))
ctx[150000, 0] = 1.0
str = np.zeros((int(T/dt), 1))
str[152200, 0] = 1.0
ctx = gaussian_filter1d(ctx, 100, axis=0)
str = gaussian_filter1d(str, 100, axis=0)

# plt.plot(ctx)
# plt.plot(str)
# plt.show()

# eic = CircuitIR.from_yaml("config/stn_gpe/stn_pop").compile(backend='numpy', step_size=dt, solver='scipy')
# results, t = eic.run(simulation_time=T, sampling_step_size=dts, profile=True,
#                      outputs={
#                          'r_e': 'stn/stn_simple/R_e',
#                          #'v_e': 'stn/qif_stn/V_e',
#                          #'v_i': 'gpe/qif_gpe/V_i',
#                               },
#                      inputs={'stn/stn_simple/u': inp}
#                      )
#
# results = results * 1e3
# results.plot()

eic2 = CircuitIR.from_yaml("config/stn_gpe/stn_gpe_basic").compile(backend='numpy', solver='scipy', step_size=dt)
results2, t = eic2.run(simulation_time=T, sampling_step_size=dts, profile=True,
                       outputs={
                           'R_e': 'stn/stn_basic/R_e',
                           'R_i': 'gpe/gpe_basic/R_i',
                                },
                       inputs={'stn/stn_basic/ctx': ctx}
                       )
# eic2.set_node_var('stn_gpe/qif_driver/delta_e', 2.0)
#eic2.generate_auto_def(None)
# print(results2.iloc[-1, :])
results2.plot()
plt.show()
