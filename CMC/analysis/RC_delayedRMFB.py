import matplotlib.pyplot as plt
from CMC.config import IFRC
import numpy as np
from copy import deepcopy

# parameters
T = 100000.0
dt = 1e-1
tau_r = 10.0
N1 = 10
N2 = 10
inp_pattern = [0, 1, 2, 3, 4, 9]
target_pattern = [0, 1, 2, 3, 4, 9]
inp_dur = int(10.0/dt)
inp_change = int(T-10000.0/dt)

# connectivities
W_intra = np.random.uniform(0., 1.0, size=(N1, N1))
#W_intra[W_intra < 0.5] = 0.
W_intra /= np.sum(W_intra, axis=1, keepdims=True)
W_inter = np.zeros((N2, N2))
W_out = np.random.uniform(0., 1.0, size=(N2, N1)) * 0.0
W_fb = np.random.randn(N1, N2) * 0.0
D_out = np.random.uniform(0.0, 10.0, size=(N2, N1))

# input and output target
steps = int(T/dt)
inp_bg_r = 440.0 + np.random.randn(steps, N1) * 40.0 * np.sqrt(dt)
inp_bg_f = 0.0 + np.random.randn(steps, N2) * 44.0 * np.sqrt(dt)
inp = np.zeros((steps, N1))
target = np.zeros((steps, max(target_pattern) + 1))
freq = 4000
step_idx = 2000
while step_idx < inp.shape[0]:
    for idx, (i, t) in enumerate(zip(target_pattern, inp_pattern)):
        target[step_idx + idx*inp_dur:step_idx + (idx+1)*inp_dur, t] = 1.0
        if step_idx < inp_change:
            inp[step_idx + idx*inp_dur:step_idx + (idx+1)*inp_dur, i] = 200.0
        else:
            inp[step_idx:step_idx + inp_dur, inp_pattern[0]] = 200.0
    step_idx += freq
net = IFRC(W_intra=deepcopy(W_intra), W_inter=W_inter, W_out=W_out, W_fb=W_fb, D_out=D_out, k=400.0,
           output_neurons=np.arange(0, max(target_pattern) + 1), tau_r=tau_r, dt=dt)
vr, vf, s = net.run(T=T, target=target, I_r=inp+inp_bg_r, I_f=inp_bg_f)
spikes = [np.argwhere(s_tmp > 0).squeeze() for s_tmp in s.T]
target_spikes = [np.argwhere(t_tmp > 0).squeeze() for t_tmp in target.T]
fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(15, 15))
axes[0, 0].plot(vr[inp_change:-1])
axes[0, 0].set_ylim(-80.0, 20.0)
axes[0, 1].plot(vf[inp_change:-1])
axes[0, 1].set_ylim(-80.0, 20.0)
try:
    axes[1, 0].eventplot(spikes[inp_change:, :], colors='k', lineoffsets=1, linelengths=1)
    axes[1, 0].set_xlim(0, steps)
except TypeError:
    pass
axes[1, 1].eventplot(target_spikes, colors='r', lineoffsets=1, linelengths=1)
axes[1, 1].set_xlim(inp_change, int(T/dt))
m1 = axes[2, 0].imshow(W_intra)
m2 = axes[2, 1].imshow(net.R.W)
fig.colorbar(m1, ax=axes[2, 0])
fig.colorbar(m2, ax=axes[2, 1])
plt.tight_layout()
plt.show()
