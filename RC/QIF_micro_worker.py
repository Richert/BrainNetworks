import sys
path = "C:\\Users\\rgf3807\\PycharmProjects\\BrainNetworks"
sys.path.append(path)
from RNNs import QIFExpAddSyns
import numpy as np
import sys
import pickle

# STEP 0: Define simulation condition
#####################################

# parse worker indices from script arguments
idx_cond = int(sys.argv[1])

# STEP 1: Load pre-generated RNN parameters
###########################################

config = pickle.load(open(f"{path}\\RC\\results\\qif_micro_config.pkl", 'rb'))

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

# input strength
w = config['alphas'][idx_cond]

# eta
eta = config['etas'][idx_cond]

# STEP 2: define remaining network parameters
#############################################

# general parameters
N = C.shape[0]
m = W_in.shape[0]
n_folds = 10
alpha = [0.001, 0.01, 0.1]

# qif parameters
Delta = 2.0
J = 15.0*np.sqrt(Delta)

# STEP 3: Evaluate classification performance of RNN
####################################################

# setup QIF RNN
qif_rnn = QIFExpAddSyns(C, eta, J, Delta=Delta, )

# perform simulation
X = qif_rnn.run(T, dt, dts, inp=inp, W_in=W_in, state_record_key='t1', cutoff=cutoff)
y = targets

# train RNN
scores = qif_rnn.kfold_crossval(X=X, y=y, k=n_folds, alphas=alpha, cv=n_folds)
avg_score = np.mean(scores, axis=0)
np.save(f"{path}\\RC\\Results\\cv_score_{idx_cond}", avg_score)
