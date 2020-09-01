import numpy as np
from pyrates.frontend import OperatorTemplate

equations = ['d/dt * r = (delta/(PI*tau) + 2.*r*v) /tau',
             'd/dt * v = (v^2 + eta + I_ext + (J*r+r_in)*tau*C - (PI*r*tau)^2) /tau']

variables = {'delta': {'default': 1.0},
             'tau': {'default': 1.0},
             'eta': {'default': -5.0},
             'J': {'default': 15.0},
             'C': {'default': 1.0},
             'r': {'default': 'output'},
             'v': {'default': 'variable'},
             'I_ext': {'default': 'input'},
             'r_in': {'default': 'input'}
            }

qif_op = OperatorTemplate(name='qif', path=None, equations=equations, variables=variables)

from pyrates.frontend import NodeTemplate, CircuitTemplate

pc = NodeTemplate(name='pc', path=None, operators=[qif_op])
ein = NodeTemplate(name='ein', path=None, operators={qif_op: {'J': 5.0, 'eta': -6.0}})
iin = NodeTemplate(name='iin', path=None, operators={qif_op: {'tau': 3.0, 'J': -5.0, 'eta': -6.0}})

qif_template = CircuitTemplate(name='net', path=None,
                               nodes={'pc': pc, 'ein': ein, 'iin': iin},
                               edges=[('pc/qif/r', 'ein/qif/r_in', None, {'weight': 5.0}),
                                      ('ein/qif/r', 'pc/qif/r_in', None, {'weight': 10.0}),
                                      ('pc/qif/r', 'iin/qif/r_in', None, {'weight': 5.0}),
                                      ('iin/qif/r', 'pc/qif/r_in', None, {'weight': -10.0})]
                              )

T = 60.
dt = 1e-3
dts = 1e-2

params = {'C': {'min': 1.0, 'max': 2.0}, 'J': {'min': 10.0, 'max': 15.0}}
param_map={'J': {'vars': ['qif/J'], 'nodes': ['pc']},
           'C': {'vars': ['qif/C'], 'nodes': ['pc', 'ein','iin']}
          }
output = {'v': 'pc/qif/v'}


def loss(data, min_amp=-10.0, max_amp=10.0, tmin=30.0):
    """Calculates the difference between the value range in the data and the
    range defined by min_amp and max_amp.
    """
    data = data.loc[tmin:, 'v']

    # calculate the difference between the membrane potential range
    # of the model and the target membrane potential range
    data_bounds = np.asarray([np.min(data), np.max(data)]).squeeze()
    target_bounds = np.asarray([min_amp, max_amp])
    diff = data_bounds - target_bounds

    # return the sum of the squared errors
    return diff @ diff.T

from pyrates.utility.genetic_algorithm import DifferentialEvolutionAlgorithm

diff_eq = DifferentialEvolutionAlgorithm()

winner = diff_eq.run(initial_gene_pool=params,
                     gene_map=param_map,
                     template=qif_template,
                     compile_kwargs={'solver': 'scipy', 'backend': 'numpy', 'step_size': dt, 'verbose': False},
                     run_kwargs={'step_size': dt, 'simulation_time': T, 'sampling_step_size': dts,
                                 'outputs': output, 'verbose': False},
                     loss_func=loss,
                     loss_kwargs={},
                     workers=-1, strategy='best2exp', mutation=(0.5, 1.9), recombination=0.8, atol=1e-3, tol=1e-2,
                     polish=False)
