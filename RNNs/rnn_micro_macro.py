import numpy as np
from rnn import QIFExpAddSyns, mQIFExpAddSynsRNN
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# parameter definitions
#######################

# simulation parameters
T = 100.0
dt = 1e-3
dts = 1e-1
cutoff = 0.0

# network configuration parameters
N = 800
p = 0.2

# setup connectivity matrix
neurons = np.arange(0, N)
C = np.random.uniform(low=1e-4, high=1, size=(N, N))
n_incoming = int(N*(1-p))
for i in range(N):
    C[np.random.choice(neurons, size=n_incoming, replace=False), i] = 0
vals, vecs = eigs(C, k=int(N/10))
sr = np.max(np.real(vals))
C /= sr

# QIF parameters
eta = -0.2
Delta = 0.3
J = 10.0
alpha = 0.3
tau_a = 10.0
tau_s = 0.5

# simulations
#############

# setup QIF RNN
qif_rnn = QIFExpAddSyns(C, eta, J, Delta=Delta, alpha=alpha, tau_s=tau_s, tau_a=tau_a, tau=1.0)

# perform simulation
results = qif_rnn.run(T, dt, dts, cutoff=cutoff, outputs=(np.arange(0, N), np.arange(3*N, 4*N)))
v_qif = np.mean(results[0], axis=1)
r_qif = np.mean(results[1], axis=1)

# setup mean-field model
C_m = np.ones(shape=(1,))
qif_mf = mQIFExpAddSynsRNN(C_m, eta, J, Delta=Delta, tau=1.0, alpha=alpha, tau_a=tau_a, tau_s=tau_s)
results = qif_mf.run(T, dt, dts, cutoff=cutoff, outputs=([0], [1]))
v_mf = np.squeeze(results[0])
r_mf = np.squeeze(results[1])

# plotting
##########

times = np.linspace(0, T, len(v_mf))
fig, axes = plt.subplots(nrows=2)

ax1 = axes[0]
ax1.plot(times, v_qif)
ax1.plot(times, v_mf)
plt.legend(['QIF', 'FRE'])
ax1.set_title('average membrane potential')
ax1.set_xlabel('time (ms)')
ax1.set_ylabel('v')

ax2 = axes[1]
ax2.plot(times, r_qif)
ax2.plot(times, r_mf)
plt.legend(['QIF', 'FRE'])
ax2.set_title('average firing rate')
ax2.set_xlabel('time (ms)')
ax2.set_ylabel('r')

plt.show()
