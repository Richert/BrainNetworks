from pyrates.ir import CircuitIR
from matplotlib.pyplot import show
T = 100.0
dt = 1e-3
dts = 1e-2

qif = CircuitIR.from_yaml("qifs/QIF_sd_lorenz")
qif_compiled = qif.compile(step_size=dt, solver='scipy')
results = qif_compiled.run(simulation_time=T, step_size=dt, sampling_step_size=dts, inputs={},
                           outputs={'r': 'qif/Op_sd_exp/r', 'x': 'inp/Op_lorenz/l1'})
results.plot()
show()
