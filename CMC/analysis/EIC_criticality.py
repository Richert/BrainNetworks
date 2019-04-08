import matplotlib.pyplot as plt
from CMC.config import EIC
import numpy as np

# parameters
############

T = 10000.0             # simulation time (s)
dt = 1e-1               # integration step-size (s)
Ne = 100                # number of excitatory neurons
Ni = 20                 # number of inhibitory neurons

# connectivities
################

W_ee = np.random.uniform(0., 1.0, size=(Ne, Ne))      # connections between excitatory neurons
W_ee = (W_ee >= 0.9) * 1.0
W_ii = np.zeros((Ni, Ni))                             # connections between inhibitory neurons
W_ei = np.random.uniform(0., 1.0, size=(Ne, Ni))      # connections from inhibitory to excitatory neurons
W_ie = np.random.uniform(0., 1.0, size=(Ni, Ne))      # connections from excitatory to inhibitory neurons

# input
steps = int(T/dt)
Ie = 500.0 + np.random.randn(steps, Ne) * 50.0 * np.sqrt(dt)   # input current for excitatory neurons
Ii = 440.0 + np.random.randn(steps, Ni) * 44.0 * np.sqrt(dt)   # input current for inhibitory neurons

# ei-circuit setup and simulation
#################################

net = EIC(W_ee=W_ee, W_ii=W_ii, W_ei=W_ei, W_ie=W_ie, k=200.0, dt=dt)    # setup of network
v, s = net.run(T=T, I_e=Ie, I_i=Ii)                                      # network simulation
spikes = [np.argwhere(s_tmp > 0).squeeze() for s_tmp in s.T]             # spike extraction

# plotting
##########

fig, axes = plt.subplots(ncols=2, figsize=(15, 8))
axes[0].plot(v[:-1])                                                      # plot membrane potentials
axes[0].set_ylim(-80.0, 20.0)
try:
    axes[1].eventplot(spikes, colors='k', lineoffsets=1, linelengths=1)   # plot spikes
    axes[1].set_xlim(0, steps)
except TypeError:
    pass
plt.tight_layout()
plt.show()
