import numpy as np
from scipy.sparse.linalg import eigs
import pickle


# simulation parameters
#######################

T = 1125.0
dt = 1e-3
dts = 1e-1
cutoff = 125.0

# network configuration parameters
##################################

N = 1500
p = 0.2
m = 5

# setup connectivity matrix
neurons = np.arange(0, N)
C = np.random.uniform(low=1e-4, high=1, size=(N, N))
n_incoming = int(N*(1-p))
for i in range(N):
    C[np.random.choice(neurons, size=n_incoming, replace=False), i] = 0
vals, vecs = eigs(C, k=int(N/10))
sr = np.max(np.real(vals))
C /= sr

# setup input matrix
p_in = 0.1
W_in = np.random.rand(N, m)
W_sorted = np.sort(W_in.flatten())
idx = W_in < W_sorted[int(N*m*p_in)]
W_in[idx] = 0.0
idx2 = W_in >= W_sorted[int(N*m*p_in)]
n_tmp = np.sum(idx2)
W_in[idx2] = np.random.uniform(-1, 1, n_tmp)
for i in range(m):
    w_sum = np.sum(W_in[:, i])
    while np.abs(w_sum) > 1e-6:
        W_in[idx2] = np.random.normal(loc=-w_sum/N, scale=0.1*np.abs(w_sum)/N, size=n_tmp)
        w_sum = np.sum(W_in[:, i])


# define network input
######################

n_epochs = 10
input_rate = 30.0/100.0

steps = int(np.round(T/dt))
store_steps = int(np.round((T - cutoff)/dts))

inp_start = int(np.round(cutoff/dt))
epoch_steps = int(np.floor((steps-inp_start)/n_epochs))
input_dur = int(np.floor(epoch_steps/m))

store_start = 0
store_epoch = int(np.floor((store_steps-store_start)/n_epochs))
store_dur = int(np.floor(store_epoch/m))

inp = np.zeros((m, steps))
targets = np.zeros((m, store_steps))

for i in range(n_epochs):
    for j in range(m):

        # generate spike train
        spike_train = np.random.poisson(input_rate, (input_dur,))

        # add spike train to input array
        idx = np.arange(inp_start+(i*m+j)*input_dur, inp_start+(i*m+j+1)*input_dur)
        inp[j, idx] = spike_train

        # store target output
        idx = np.arange(store_start+(i*m+j)*store_dur, store_start+(i*m+j+1)*store_dur)
        targets[j, idx] = 1.0

import matplotlib.pyplot as plt
plt.imshow(inp, aspect='auto', interpolation='none')
plt.colorbar()
plt.show()

# store data
data = {}
data['T'] = T
data['dt'] = dt
data['dts'] = dts
data['cutoff'] = cutoff
data['N'] = N
data['p'] = p
data['C'] = C
data['W_in'] = W_in
data['inp'] = inp
data['targets'] = targets.T
data['number_input_channels'] = m
pickle.dump(data, open('data/qif_input_config.pkl', 'wb'))
