from pyrates.ir import CircuitIR
import matplotlib.pyplot as plt
import os

# parameters
dt = 1e-2
T = 5000.0
dts = 1.0

eic = CircuitIR.from_yaml("config/stn_gpe/stn_pop").compile(backend='fortran', step_size=dt, solver='scipy',
                                                            auto_compat=True)
results, t = eic.run(simulation_time=T, sampling_step_size=dts, profile=True, outputs={'r_e': 'stn/stn_simple/R_e'})
eic.generate_auto_def(f"{os.getcwd()}/config")
results.plot()
plt.show()
