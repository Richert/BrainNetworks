from pyrates.utility import grid_search_annarchy, plot_timeseries
from ANNarchy import Projection, Population, TimedArray, setup, Network, Monitor, Uniform, Normal, \
    EIF_cond_exp_isfa_ista
from pyrates.utility import pyrates_from_annarchy
import matplotlib.pyplot as plt
import numpy as np

# parameters
############

T = 1000.0              # simulation time (ms)
dt = 1e-2               # integration step-size (ms)
Ne = 100                # number of excitatory neurons
Ni = 100                # number of inhibitory neurons
c_min = 0.1
c_max = 1.0

# network definition
####################

setup(method='explicit', dt=dt)

# Neuron definition
neuron = EIF_cond_exp_isfa_ista()
neuron.equations = """    
    I = g_exc * (e_rev_E - v) + g_inh * (e_rev_I - v) + i_offset * Normal(0.2, 1.0)

    tau_m * dv/dt = (v_rest - v +  delta_T * exp((v-v_thresh)/delta_T)) + tau_m/cm*(I - w) : init=-70.6

    tau_w * dw/dt = a * (v - v_rest) / 1000.0 - w 

    tau_syn_E * dg_exc/dt = - g_exc : exponential
    tau_syn_I * dg_inh/dt = - g_inh : exponential
"""

# population setup
pop = Population(Ne + Ni, neuron=neuron)
E = pop[:Ne]
I = pop[Ne:]

# projection setup
C_ei = Projection(pre=E, post=I, target='exc', name='EI')
C_ie = Projection(pre=I, post=E, target='inh', name='IE')
#C_ee = Projection(E, E, 'exc', name='EE')
#C_ii = Projection(I, I, 'inh', name='II')
C_ei.connect_fixed_probability(0.1, weights=Uniform(c_min, c_max))
C_ie.connect_fixed_probability(0.1, weights=Uniform(c_min, c_max))
#C_ee.connect_fixed_probability(0.3, weights=Uniform(c_min, c_max))
#C_ii.connect_fixed_probability(0.3, weights=Uniform(c_min, c_max))

# input
#steps = int(T/dt)
#I_e_tmp = 5.0 + np.random.randn(steps, Ne) * 50.0 * np.sqrt(dt)   # input current for excitatory neurons
#I_i_tmp = 4.0 + np.random.randn(steps, Ni) * 44.0 * np.sqrt(dt)   # input current for inhibitory neurons
#I_e = TimedArray(rates=I_e_tmp, name="E_inp")
#I_i = TimedArray(rates=I_i_tmp, name="I_inp")
#inp_e = Projection(pre=I_e, post=E, target='exc')
#inp_i = Projection(pre=I_i, post=I, target='exc')
#inp_e.connect_one_to_one(1.0)
#inp_i.connect_one_to_one(1.0)
E.i_offset = 5.0
I.i_offset = 2.0

# monitoring
obs_e = Monitor(E, variables=['spike', 'v'], start=True)
obs_i = Monitor(I, variables=['spike', 'v'], start=True)

# simulation
############

# annarchy simulation
net = Network(everything=True)
net.compile()
net.simulate(duration=T)

# conversion to pyrates
rate_e = pyrates_from_annarchy(monitors=[net.get(obs_e)], vars=['spike'], pop_average=True)
rate_i = pyrates_from_annarchy(monitors=[net.get(obs_i)], vars=['spike'], pop_average=True)
v_e = pyrates_from_annarchy(monitors=[net.get(obs_e)], vars=['v'], pop_average=False)
v_i = pyrates_from_annarchy(monitors=[net.get(obs_i)], vars=['v'], pop_average=False)

# visualization
###############

plt.plot(rate_e)
plt.plot(rate_i)
plt.figure()
plt.plot(v_e)
plt.figure()
plt.plot(v_i)
plt.show()
