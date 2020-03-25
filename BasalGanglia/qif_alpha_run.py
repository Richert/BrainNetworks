from pyrates.utility import plot_timeseries, grid_search, plot_psd, plot_connectivity
from pyrates.ir import CircuitIR
import numpy as np
import matplotlib.pyplot as plt

# parameters
dt = 1e-4
T = 1000.0
dts = 1e-1

eic = CircuitIR.from_yaml("config/stn_gpe/qif_alpha_net").compile(backend='numpy', solver='scipy', step_size=dt)
results, t = eic.run(simulation_time=T, sampling_step_size=dts, profile=True,
                     outputs={'Re': 'pop/qif_single_alpha/R_e'})

print(results.iloc[-1, :])
results.plot()
plt.show()
