from rnn import QIFExpAddSyns
from rnn import mQIFExpAddSynsRNN
import numpy as np
import pickle
from scipy.ndimage import gaussian_filter1d


def kuramoto_order_parameter(r, v):
    W = np.asarray([complex(np.pi * r_tmp, v_tmp) for r_tmp, v_tmp in zip(r, v)])
    W_c = W.conjugate()
    return np.abs((1 - W_c) / (1 + W_c))


# STEP 0: Define simulation condition
#####################################

# parse worker indices from script arguments
idx_cond = 1

# STEP 1: Load pre-generated RNN parameters
###########################################

config = pickle.load(open("data/qif_input_config.pkl", 'rb'))

# connectivity matrix
C = config['C']
N = C.shape[0]

# QIF input
inp = config['inp']

# simulation config
T = config['T']
dt = config['dt']
dts = config['dts']
cutoff = config['cutoff']
t = int((T - cutoff)/dts)
m = config['number_input_channels']

# target values
y = config['targets']

# mean-field input
start = int(np.round(0.5*T/dt))
stop = start + int(np.round(1/dt))
inp_mf = np.zeros((1, int(np.round(T/dt))))
inp_mf[0, start:stop] = 1.0

# STEP 1: define remaining network parameters
#############################################

# general parameters
n_folds = 5
ridge_alpha = 0.5*10e-3

# qif parameters
Delta = 0.3
J = 7.5
eta = -0.6
tau_a = 10.0
tau_s = 1.0

# adaptation strength
alpha = 0.3

# independent variable (IV)
iv_name = "p_in"
ivs = [0.5, 0.6, 0.7]
n_iv = len(ivs)

# mean-field parameters
C_m = np.ones(shape=(1,))

# STEP 3: Evaluate classification performance of RNN
####################################################

data = dict()
data["score"] = np.zeros((n_iv,))
data["wta_score"] = np.zeros_like(data["score"])
data["r_qif"] = np.zeros((n_iv, t))
data["v_qif"] = np.zeros_like(data["r_qif"])
data["r_mf"] = np.zeros_like(data["r_qif"])
data["v_mf"] = np.zeros_like(data["r_qif"])
data["Z_qif"] = np.zeros_like(data["r_qif"])
data["Z_mf"] = np.zeros_like(data["r_qif"])
data["iv"] = ivs
data["iv_name"] = iv_name

# simulation loop for n_etas
for j in range(n_iv):

    iv = ivs[j]
    print(f'Performing simulations for {iv_name} = {iv} ...')

    # setup input matrix
    p_in = iv
    W_in = np.random.rand(N, m)
    W_sorted = np.sort(W_in.flatten())
    idx = W_in > W_sorted[int(N * m * p_in)]
    idx2 = W_in <= W_sorted[int(N * m * p_in)]
    W_in[idx] = 0.0
    for i in range(m):
        indices = np.argwhere(idx2[:, i]).squeeze().tolist()
        np.random.shuffle(indices)
        n_half = int(len(indices) / 2)
        while len(indices) > 1:
            w = np.random.uniform(-1, 1)
            idx_tmp1 = indices.pop()
            idx_tmp2 = indices.pop()
            W_in[idx_tmp1, i] = w
            W_in[idx_tmp2, i] = -w
        if len(indices) == 1:
            W_in[indices.pop(), i] = 0.0
    print(np.sum(W_in, axis=0))

    # setup QIF RNN
    qif_rnn = QIFExpAddSyns(C, eta, J=J, Delta=Delta, alpha=alpha, tau_s=tau_s, tau_a=tau_a, tau=1.0)

    # perform simulation
    results = qif_rnn.run(T, dt, dts, inp=inp, W_in=W_in, cutoff=cutoff, outputs=(np.arange(0, N), np.arange(3*N, 4*N)),
                          verbose=False)
    v_qif = np.mean(results[0], axis=1)
    r_qif = np.mean(results[1], axis=1)
    X = results[1]

    # prepare training data
    buffer_val = 0
    for i in range(X.shape[1]):
        X[:, i] = gaussian_filter1d(X[:, i], 0.1/dts, mode='constant', cval=buffer_val)
    r_qif2 = np.mean(X, axis=1)

    # split into test and training data
    split = int(np.round(X.shape[0]*0.75, decimals=0))
    X_train = X[:split, :]
    y_train = y[:split, :]
    X_test = X[split:, :]
    y_test = y[split:, :]

    # train RNN
    key, scores, coefs = qif_rnn.ridge_fit(X=X_train, y=y_train, alpha=ridge_alpha, k=0, fit_intercept=False,
                                           copy_X=True, solver='lsqr', readout_key=f'qif_{iv_name}_{m}', verbose=False)

    # calculate classification score on test data
    score, y_predict = qif_rnn.test(X=X_test, y=y_test, readout_key=key)

    # Winner takes it all classification
    wta_pred = y_predict.argmax(axis=1)
    wta_target = y_test.argmax(axis=1)
    wta_score = np.mean(wta_pred == wta_target)

    # simulate mean-field dynamics
    qif_mf = mQIFExpAddSynsRNN(C_m, eta, J=J, Delta=Delta, alpha=alpha, tau_a=tau_a, tau_s=tau_s, tau=1.0)
    results = qif_mf.run(T, dt, dts, cutoff=cutoff, outputs=([0], [1]), inp=inp_mf, W_in=np.ones((1, 1)))
    v_mf = np.squeeze(results[0])
    r_mf = np.squeeze(results[1])

    # calculate Kuramoto order parameter Z for QIF network and mean-field model
    Z_qif = kuramoto_order_parameter(r_qif, v_qif)
    Z_mf = kuramoto_order_parameter(r_mf, v_mf)

    print(f"Finished. Results: WTA = {wta_score}, mean(Z) = {np.mean(Z_qif)}.")

    # store data
    data["score"][j] = score
    data["wta_score"][j] = wta_score
    data["r_qif"][j, :] = r_qif2
    data["v_qif"][j, :] = v_qif
    data["r_mf"][j, :] = r_mf
    data["v_mf"][j, :] = v_mf
    data["Z_qif"][j, :] = Z_qif
    data["Z_mf"][j, :] = Z_mf

data["T"] = T
pickle.dump(data, open('data/qif_rc_multichannel_results.pkl', 'wb'))
