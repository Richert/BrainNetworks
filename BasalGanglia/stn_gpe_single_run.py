from pyrates.utility import plot_timeseries, grid_search, plot_psd, plot_connectivity
from pyrates.ir import CircuitIR
import numpy as np
import matplotlib.pyplot as plt


# parameters
dt = 1e-2
T = 1000.0
dts = 1.0
inp = np.zeros((int(T/dt), 1))
inp[2000:3000] = 2.0

eic = CircuitIR.from_yaml("config/stn_gpe/stn_pop").compile(backend='numpy', step_size=dt, solver='scipy')
results, t = eic.run(simulation_time=T, sampling_step_size=dts, profile=True,
                     outputs={
                         'r_e': 'stn/stn_simple/R_e',
                         #'v_e': 'stn/qif_stn/V_e',
                         #'v_i': 'gpe/qif_gpe/V_i',
                              },
                     inputs={'stn/stn_simple/u': inp}
                     )

results = results * 1e3
results.plot()

# eic2 = CircuitIR.from_yaml("config/stn_gpe/net_stn_gpe").compile(backend='numpy', step_size=dt, solver='scipy',
#                                                                  )
# results2, t = eic2.run(simulation_time=T, sampling_step_size=dts, profile=True,
#                        outputs={
#                            'R_e': 'stn_gpe/qif_driver/R_e',
#                            'R_i': 'stn_gpe/qif_driver/R_i',
#                                 },
#                        )
# eic2.set_node_var('stn_gpe/qif_driver/delta_e', 2.0)
#eic2.generate_auto_def(None)
# print(results2.iloc[-1, :])
# results2.plot()
plt.show()
