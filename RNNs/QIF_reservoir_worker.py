from RNNs import QIFExpAddRNN
import numpy as np
import sys

# STEP 0: Define simulation condition
#####################################

# parse worker indices from script arguments
idx_w = sys.argv[1]
idx_eta = sys.argv[2]

# STEP 1: Load pre-generated RNN parameters
###########################################

path = ""

# connectivity matrix
C = np.load(f"{path}/C.npy")

# input
inp = np.load(f"{path}/inp.npy")

# input weights
W_in = np.load(f"{path}/W_in.npy")

# simulation time-steps
time = np.load(f"{path}/time.npy")

# target values
targets = np.load(f"{path}/targets.npy")

# input strength
w = np.load(f"{path}/ws.npy")[idx_w]

# eta
eta = np.load(f"{path}/etas.npy")[idx_eta]

# STEP 2: define remaining network parameters
#############################################

# general parameters
N = C.shape[0]
m = W_in.shape[0]
T = time[-1]
dt = time[1] - time[0]
dts = 1e-2
n_folds = 10
cutoff = 10.0
alpha = [0.1, 1.0, 10.0]

# qif parameters
Delta = 2.0
J = 15.0*np.sqrt(Delta)

# STEP 3: Evaluate classification performance of RNN
####################################################

# setup QIF RNN
qif_rnn = QIFExpAddRNN(C, eta, J)

# train RNN
qif_rnn.train_cv(dt, dts, targets, inp=inp, W_in=W_in, key='t1', alpha=alpha, cutoff=cutoff, n_folds=n_folds)

# store best average score (averaged over cv folds; best over alphas)
scores = qif_rnn.test('t1', dt, dts, targets, inp=inp, W_in=W_in, cutoff=cutoff)
avg_scores = np.mean(scores, axis=1)
best_score = np.max(avg_scores)
np.save(f"{path}/results/cv_score_{idx_w}_{idx_eta}", best_score)
