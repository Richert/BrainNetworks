from RNNs import QIFExpAddRNN
import numpy as np
import matplotlib.pyplot as plt

N = 2000
p = 0.05
m = 4
T = 100.0
dt = 1e-4
dts = 1e-2

# connectivity
C = np.random.rand(N, N)
C_sorted = np.sort(C.flatten())
C = 1. * (C <= C_sorted[int(N**2*p)])
C /= N*p

# extrinsic input
steps = int(np.round(T/dt))
store_steps = int(np.round(T/dts))
inp = np.zeros((m, steps))
targets = np.zeros((store_steps,))
inp_vals = [4.0, 4.0]
inp_indices = [0, 2]
inp_dur = int(np.round(5.0/dt))
inp_times = [10.0, 30.0, 50.0, 70.0]
for t in inp_times:
    start = int(np.round(t / dt))
    start_t = int(np.round(t / dts))
    for val, idx in zip(inp_vals, inp_indices):
        inp[idx, start:start+inp_dur] = val
    targets[start_t:start_t+inp_dur] = 1.0
W_in = np.random.rand(N, m)
W_sorted = np.sort(W_in.flatten())
W_in[np.abs(W_in) < W_sorted[int(N*m*0.5)]] = 0.0

# qif parameters
Delta = 1.0
eta = -4.0*Delta
J = 15.0*np.sqrt(Delta)

# setup QIF RNN
qif_rnn = QIFExpAddRNN(C, eta, J)

# train RNN
qif_rnn.train(dt, dts, targets, inp=inp, W_in=W_in, key='t1', alpha=1.0, cutoff=10.0)
score = qif_rnn.test('t1', dt, dts, targets, inp=inp, W_in=W_in, cutoff=10.0)
print(f'classification score: {score}')

# plotting
fig, axes = plt.subplots(nrows=3)
ax = axes[0]
ax.plot(np.mean(qif_rnn.fr_record[:, :N], axis=1))
ax = axes[1]
ax.plot(np.mean(qif_rnn.fr_record[:, N:], axis=1))
ax = axes[2]
ax.plot(qif_rnn.fr_record[:, :N])
plt.tight_layout()
plt.show()
