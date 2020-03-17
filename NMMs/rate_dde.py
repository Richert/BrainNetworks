from pyrates.utility import grid_search
import numpy as np
import matplotlib.pyplot as plt

# parameters and inputs
T = 200.0
dt = 0.001
steps = int(np.round(T/dt, decimals=0))
u = np.zeros((steps,1))
u[100:200] = 10.0
dts = 0.01

results1, _ = grid_search("rate_dde/net_dde",
                          param_grid={'k': [0.8, 0.9, 1.0, 1.1, 1.2]},
                          param_map={'k': {'vars': ['rate_dde/k'], 'nodes': ['n1']}},
                          inputs={},
                          outputs={'r': 'n1/rate_dde/r'},
                          dt=dt, simulation_time=T, sampling_step_size=dts,
                          init_kwargs={'backend': 'numpy', 'solver': 'euler', 'step_size': dt},
                          )
results1.plot()

results2, _ = grid_search("rate_dde/net_ode",
                          param_grid={'k': [0.8, 0.9, 1.0, 1.1, 1.2]},
                          param_map={'k': {'vars': ['rate_ode/k'], 'nodes': ['n1']}},
                          inputs={},
                          outputs={'r': 'n1/rate_ode/r'},
                          dt=dt, simulation_time=T, sampling_step_size=dts,
                          init_kwargs={'backend': 'numpy', 'solver': 'scipy', 'step_size': dt})
results2.plot()
plt.show()
