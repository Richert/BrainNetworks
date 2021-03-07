from pyrates.frontend import CircuitTemplate
import numpy as np
import matplotlib.pyplot as plt
from pyrates.utility.visualization import plot_network_graph
from pyrates.utility.genetic_algorithm import DifferentialEvolutionAlgorithm
from pyrates.utility.grid_search import grid_search
from pyrates.utility.visualization import Interactive2DParamPlot

# parameter definition
dt = 1e-3
dts = 1e-2
cutoff = 100.0
T = 150.0 + cutoff
start = int((0 + cutoff)/dt)
dur = int(5/(0.6*dt))
steps = int(T/dt)
inp = np.zeros((steps, 1))
inp[start:start+dur] = 0.6

# model setup
path = "../config/spinal_cord/sc"
template = CircuitTemplate.from_yaml(path).apply()
plot_network_graph(template)

model = template.compile(backend='numpy', solver='scipy', step_size=dt)

# simulation
results = model.run(simulation_time=T, step_size=dt, sampling_step_size=dts,
                    inputs={'m1/m1_dummy/m_in': inp},
                    outputs={
                        'psp': 'muscle/muscle_op/I_acc',
                        'alpha_exc': 'alpha/alpha_op/I_ampa',
                        'alpha_inh': 'alpha/alpha_op/I_glycin',
                        'renshaw': 'renshaw/renshaw_op/I_acc'
                    }
                    )

results.plot()
plt.show()
#
# # optimization
# targets = results

diff_eq = DifferentialEvolutionAlgorithm()


# def loss(data):
#     diff = data - targets
#     return diff.T @ diff


# params = {'s_alpha': {'min': 0.1, 'max': 10.0},
#           's_renshaw': {'min': 0.1, 'max': 10.0},
#           'thr_alpha': {'min': -20.0, 'max': 20.0},
#           'thr_renshaw': {'min': -20.0, 'max': 20.0},
#           'c_ra': {'min': 0.1, 'max': 100.0},
#           'c_ar': {'min': 0.03, 'max': 30.0},
#           }
v1 = 'c_ra'
v2 = 'c_ar'
params = {v1: np.linspace(1.0, 30.0, 10),
          v2: np.linspace(1.0, 30.0, 10),
          }
param_map = {'s_alpha': {'vars': ['alpha_op/s'], 'nodes': ['alpha']},
             's_renshaw': {'vars': ['renshaw_op/s'], 'nodes': ['renshaw']},
             'thr_alpha': {'vars': ['alpha_op/I_thr'], 'nodes': ['alpha']},
             'thr_renshaw': {'vars': ['renshaw_op/I_thr'], 'nodes': ['renshaw']},
             'c_ac': {'vars': ['weight'], 'edges': [('m1', 'alpha')]},
             'c_ra': {'vars': ['weight'], 'edges': [('alpha', 'renshaw')]},
             'c_ar': {'vars': ['weight'], 'edges': [('renshaw', 'alpha')]},
             'c_ma': {'vars': ['weight'], 'edges': [('alpha', 'muscle')]},
             }

# winner = diff_eq.run(initial_gene_pool=params,
#                      gene_map=param_map,
#                      template="../config/spinal_cord/sc",
#                      compile_kwargs={'solver': 'scipy', 'backend': 'numpy', 'step_size': dt, 'verbose': False},
#                      run_kwargs={'step_size': dt, 'simulation_time': T, 'sampling_step_size': dts,
#                                  'inputs': {'m1/m1_dummy/m_in': inp},
#                                  'outputs': {'psp': 'muscle/muscle_op/I_acc'},
#                                  'verbose': False},
#                      loss_func=loss,
#                      loss_kwargs={},
#                      strategy='best2exp', mutation=(0.5, 1.9), recombination=0.8, atol=1e-5, tol=1e-3,
#                      polish=False, disp=True, verbose=False, workers=-1)

# grid search

results, result_map = grid_search(
        circuit_template=path,
        param_grid=params,
        param_map=param_map,
        simulation_time=T,
        step_size=dt,
        permute_grid=True,
        sampling_step_size=dts,
        inputs={'m1/m1_dummy/m_in': inp},
        outputs={'psp': 'muscle/muscle_op/I_acc'},
        init_kwargs={'backend': 'numpy', 'solver': 'scipy', 'step_size': dt},
        method='RK45',
    )
results = results.loc[:, 'psp']
results.plot()
plt.show()

data = np.zeros((len(params[v1]), len(params[v2])))
Interactive2DParamPlot(data, results, x_values=params[v1], y_values=params[v2], param_map=result_map,
                       tmin=cutoff, x_key=v1, y_key=v2)
plt.show()
