import numpy as np
from rnn import QIFExpAddSyns, mQIFExpAddSynsRNN
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
import pickle

# parameter definitions
#######################

# simulation parameters
T = 300.0
dt = 1e-3
dts = 1e-1
cutoff = 0.0

# impuls definition
start = int(np.round(50.0/dt))
stop = int(np.round(51.0/dt))
steps = int(np.round(T/dt))
inp = np.zeros((1, steps))
inp[0, start:stop] = 1.0

# network configuration parameters
config = pickle.load(open("data/qif_input_config.pkl", 'rb'))

# setup connectivity matrix
C = config['C']
N = C.shape[0]

# setup input matrix
W_in = config['W_in']

# define network input
######################

m = 5
n_epochs = 1
input_rate = 30.0/100.0

steps = int(np.round(T/dt))
store_steps = int(np.round((T - cutoff)/dts))

inp_start = int(np.round(cutoff/dt))
epoch_steps = int(np.floor((steps-inp_start)/n_epochs))
input_dur = int(np.floor(epoch_steps/m))

store_start = 0
store_epoch = int(np.floor((store_steps-store_start)/n_epochs))
store_dur = int(np.floor(store_epoch/m))

inp_qif = np.zeros((m, steps))

for i in range(n_epochs):
    for j in range(m):
        spike_train = np.random.poisson(input_rate, (input_dur,))
        idx = np.arange(inp_start+(i*m+j)*input_dur, inp_start+(i*m+j+1)*input_dur)
        inp_qif[j, idx] = spike_train

# QIF parameters
eta = -0.6
Delta = 0.3
J = 8.4
alpha = 0.3
tau_a = 10.0
tau_s = 1.0

# simulations
#############

# setup QIF RNN
qif_rnn = QIFExpAddSyns(C, eta, J, Delta=Delta, alpha=alpha, tau_s=tau_s, tau_a=tau_a, tau=1.0)

# perform simulation
results = qif_rnn.run(T, dt, dts, cutoff=cutoff, outputs=(np.arange(0, N), np.arange(3*N, 4*N)),
                      inp=inp_qif, W_in=W_in)
v_qif = np.mean(results[0], axis=1)
r_qif = np.mean(results[1], axis=1)

# setup mean-field model
C_m = np.ones(shape=(1,))
qif_mf = mQIFExpAddSynsRNN(C_m, eta, J, Delta=Delta, tau=1.0, alpha=alpha, tau_a=tau_a, tau_s=tau_s)
results = qif_mf.run(T, dt, dts, cutoff=cutoff, outputs=([0], [1]), inp=inp, W_in=np.ones((1, 1)))
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
