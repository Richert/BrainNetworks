from RNNs import QIFExpAddNoiseSTDP
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# STEP 0: Define simulation condition
#####################################

# parse worker indices from script arguments
idx_cond = 570

# STEP 1: Load pre-generated RNN parameters
###########################################

path = "/home/rgast/PycharmProjects/BrainNetworks/RC/results"
config = pickle.load(open(f"{path}/qif_micro_config.pkl", 'rb'))

# connectivity matrix
C = config['C']

# input
inp = config['inp']

# input weights
W_in = config['W_in']

# simulation config
T = config['T']
dt = config['dt']
dts = config['dts']
cutoff = config['cutoff']

# target values
targets = config['targets']

# adaptation strength
alpha = 0.2  # config['alphas'][idx_cond]

# eta
eta = -0.8  # config['etas'][idx_cond]

# STEP 2: define remaining network parameters
#############################################

# general parameters
N = C.shape[0]
m = W_in.shape[0]
n_folds = 5
ridge_alpha = 1e-5

# qif parameters
Delta = 0.1
J = 6.2
D = 0.2

# save original weight matrix
C_old = np.zeros_like(C)
C_old[:] = C[:]

# STEP 3: Evaluate classification performance of RNN
####################################################

# setup QIF RNN
qif_rnn = QIFExpAddNoiseSTDP(C, eta, J, Delta=Delta, alpha=alpha, D=D, beta=0.1, gamma_p=5e-3, gamma_n=1e-2, tau_e=0.5)

# perform simulation
X = qif_rnn.run(T, dt, dts, inp=inp, W_in=W_in, state_record_key='t1', cutoff=cutoff)

# prepare training data
buffer_val = 0
for i in range(X.shape[1]):
    X[:, i] = gaussian_filter1d(X[:, i], 1.0 / dts, mode='constant', cval=buffer_val)
y = targets

# train RNN
key, scores, coefs = qif_rnn.ridge_fit(X=X, y=y, alpha=ridge_alpha, k=n_folds, fit_intercept=False, copy_X=True,
                                       solver='lsqr')
score, y_predict = qif_rnn.test(X=X, y=y, readout_key=key)

print(f'classification score: {score}')

# plotting
fig, axes = plt.subplots(nrows=4)

ax1 = axes[0]
ax1.plot(np.mean(X, axis=1))

ax2 = axes[1]
im = ax2.imshow(X.T, aspect='auto', cmap="plasma", vmin=0, vmax=0.01)
#plt.colorbar(im, ax=ax2, shrink=0.5)

ax3 = axes[2]
ax3.plot(y)
ax3.plot(y_predict)
plt.legend(['target', 'output'])

ax4 = axes[3]
start = int(cutoff/dt)
ax4.plot(inp[0, start:])
ax4.plot(inp[1, start:])
plt.legend(['lorenz', 'stula'])

plt.tight_layout()

# plot connectivities
fig2, axes2 = plt.subplots(ncols=2)

ax1 = axes2[0]
im1 = ax1.imshow(C_old, aspect='auto', cmap="plasma", vmin=0, vmax=np.max(C[:]))
plt.colorbar(im1, ax=ax1, shrink=0.5)
plt.title('pre-STDP')

ax2 = axes2[1]
im2 = ax2.imshow(C, aspect='auto', cmap="plasma", vmin=0, vmax=np.max(C[:]))
plt.colorbar(im2, ax=ax2, shrink=0.5)
plt.title('post-STDP')

plt.tight_layout()

print(f'Original synaptic sparseness: {np.sum(C_old[:] == 0)/N**2}')
print(f'New synaptic sparseness: {np.sum(C[:] == 0)/N**2}')
plt.show()
