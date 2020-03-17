from pyrates.utility import grid_search_annarchy, plot_timeseries
from ANNarchy import Projection, Population, Neuron, setup, Monitor, compile, simulate
from pyrates.utility import pyrates_from_annarchy
import matplotlib.pyplot as plt
import numpy as np

# Parameters and initial conditions
N = 1000
D = 2.0
eta = -5.0*D
J = 15.0*np.sqrt(D)
tau = 10.0
tau_syn = 0.1
v_th = 100.0
T = 50.0
dt = 0.001
setup(method='explicit', dt=dt)

# theta neuron definition
Theta = Neuron(
    parameters=f"""
        tau = {tau}  : population
        eta = {eta}
        J = {J} : population
        tau_s = {tau_syn} : population
    """,
    equations="""
        v_old = v_new
        tau * dv/dt = 1.0 - cos(v_old) + (1.0 + cos(v_old))*(eta + J*g_syn*tau) : init=6.2832, min=0.0
        tau_s * dg_syn/dt = g_exc*tau_s - g_syn
        v_tmp = v/(2*pi) : int
        v_new = (v/(2*pi) - v_tmp)*2*pi
    """,
    spike="(v_new > pi)*(v_old < pi)"
)

# population setup
pop1 = Population(N, neuron=Theta, name="ThetaPop1")
pop1.eta = eta + D*np.tan((np.pi/2)*(2*np.arange(1, N+1)-N-1)/(N+1))

# projection setup
proj = Projection(pre=pop1, post=pop1, target='exc', name='fb')
proj.connect_all_to_all(100.0/N, allow_self_connections=False)

# monitoring
obs = Monitor(pop1, variables=['spike', 'v_new'], start=True, period=0.01)

# simulation
compile()
simulate(duration=T)

# conversion to pyrates
theta = pyrates_from_annarchy(monitors=[obs], vars=['v_new'], pop_average=False)
rate = pyrates_from_annarchy(monitors=[obs], vars=['spike'], pop_average=True)

plt.plot(theta)
plt.figure()
plt.plot(rate)
plt.show()
