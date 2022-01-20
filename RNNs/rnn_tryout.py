from RNNs import QIFExpAddSyns
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

path = "C:\\Users\\rgf3807\\PycharmProjects\\BrainNetworks\\RC\\Results"
config = pickle.load(open(f"{path}\\qif_input_config.pkl", 'rb'))

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
alpha = 0.5  # config['alphas'][idx_cond]

# eta
eta = -3.8  # config['etas'][idx_cond]

# STEP 2: define remaining network parameters
#############################################

# general parameters
N = C.shape[0]
m = W_in.shape[1]
n_folds = 5
ridge_alpha = 1e-3

# qif parameters
Delta = 2.0
J = 15.0*np.sqrt(Delta)
tau_a = 10.0
tau_s = 0.8

# STEP 3: Evaluate classification performance of RNN
####################################################

# setup QIF RNN
qif_rnn = QIFExpAddSyns(C, eta, J, Delta=Delta, alpha=alpha, tau_s=tau_s, tau_a=tau_a)

# perform simulation
W_in[:, :] = 0.0
X = qif_rnn.run(T, dt, dts, inp=inp, W_in=W_in, cutoff=cutoff)[0]
r_qif = np.mean(X, axis=1)

# prepare training data
buffer_val = 0
for i in range(X.shape[1]):
    X[:, i] = gaussian_filter1d(X[:, i], 0.05 / dts, mode='constant', cval=buffer_val)
y = targets
r_qif2 = np.mean(X, axis=1)

# split into test and training data
split = int(np.round(X.shape[0]*0.75, decimals=0))
X_train = X[:split, :]
y_train = y[:split]
X_test = X[split:, :]
y_test = y[split:]

# train RNN
key, scores, coefs = qif_rnn.ridge_fit(X=X_train, y=y_train, alpha=ridge_alpha, k=n_folds, fit_intercept=False, copy_X=True,
                                       solver='lsqr')
score, _ = qif_rnn.test(X=X_test, y=y_test, readout_key=key)
y_predict = qif_rnn.predict(X=X, readout_key=key)
print(f"Classification performance on test data: {score}")

# plotting
fig, axes = plt.subplots(nrows=4)

ax1 = axes[0]
ax1.plot(np.mean(X, axis=1))

ax2 = axes[1]
im = ax2.imshow(X.T, aspect='auto', cmap="plasma", vmin=0, vmax=0.005)
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

# plot connectivity
fig2, ax = plt.subplots()

im1 = ax.imshow(C, aspect='auto', cmap="plasma", vmin=0, vmax=np.max(C[:]))
plt.colorbar(im1, ax=ax, shrink=0.5)
plt.title('C')

plt.tight_layout()
print(f'Synaptic sparseness: {np.sum(C[:] == 0)/N**2}')
plt.show()
