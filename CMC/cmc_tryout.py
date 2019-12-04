from pyrates.utility import plot_timeseries, grid_search
from pyrates.frontend import CircuitTemplate
import numpy as np
import matplotlib.pyplot as plt

# parameters
dt = 1e-4
dts = 1e-3
T = 1.0
u1 = np.random.uniform(120, 320, (int(T/dt), 1))
u2 = np.random.uniform(120, 320, (int(T/dt), 1)) * 0.0

# initialization
cmc = CircuitTemplate.from_yaml("model_templates.jansen_rit.simple_jansenrit.JRC_2coupled").apply()
compute_graph = cmc.compile(backend='numpy', dt=dt)

# simulation
results = compute_graph.run(simulation_time=T,
                            outputs={'V1': 'JRC1/JRC_op/PSP_pc', 'V2': 'JRC2/JRC_op/PSP_pc'},
                            inputs={'JRC2/JRC_op/u': u1})

# plotting
results.plot()
plt.show()
