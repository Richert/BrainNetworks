from rnn import QIFExpAddSyns
from rnn import kuramoto_order_parameter, create_connectivity_matrix
import numpy as np
import pickle
from scipy.sparse.linalg import eigs


def simulate(N: int, p: float, eta: float, J: float, Delta: float, alpha: float, tau_s: float, tau_a: float,
             T: float, dt: float, dts: float, inp: np.ndarray, W_in: np.ndarray, cutoff: float):

    # setup connectivity matrix
    C = create_connectivity_matrix(N, p)

    # setup QIF RNN
    qif_rnn = QIFExpAddSyns(C, eta, J=J, Delta=Delta, alpha=alpha, tau_s=tau_s, tau_a=tau_a, tau=1.0)

    # perform simulation
    results = qif_rnn.run(T, dt, dts, inp=inp, W_in=W_in, cutoff=cutoff,
                          outputs=(np.arange(0, N), np.arange(3 * N, 4 * N)),
                          verbose=False)
    r_qif = np.mean(results[1], axis=1)
    return np.max(r_qif), C

# STEP 0: Define simulation condition
#####################################

# parse worker indices from script arguments
idx_cond = 1

# STEP 1: Load pre-generated RNN parameters
###########################################

config = pickle.load(open("data/qif_fit_macro_config.pkl", 'rb'))

# connectivity matrix
C = config['C']

# QIF input
inp = config['inp']

# input weights
W_in = config['W_in']

# simulation config
dt = config['dt']
dts = config['dts']
cutoff = config['cutoff']
T = config['T']
t = int((T - cutoff) / dts)
M = config['number_input_channels']

# target values
y = config['targets']

# STEP 1: define remaining network parameters
#############################################

# general parameters
N = C.shape[0]
m = W_in.shape[0]
n_folds = 5
ridge_alpha = 0.5 * 10e-3

# qif parameters
Delta = 0.3
eta = -0.6
J = 8.35
tau_a = 10.0
tau_s = 1.0

# adaptation strength
alpha = 0.3

# coupling probability
p = 0.05

# STEP 3: Evaluate mean-field dynamics of RNN
#############################################

max_rate = 0.6
attempt = 0
while max_rate > 0.45 and attempt < 100:
    max_rate, C = simulate(N, p, eta, J, Delta, alpha, tau_s, tau_a, T, dt, dts, inp, W_in, cutoff)
    attempt += 1
    print(f'attempt #{attempt}: max_rate = {max_rate}')

config["C"] = C
pickle.dump(config, open('data/qif_fit_macro_config.pkl', 'wb'))
