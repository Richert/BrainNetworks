from rnn import QIFExpAddSyns
from rnn import mQIFExpAddSynsRNN
from rnn import kuramoto_order_parameter
import numpy as np
import pickle


# STEP 0: Define simulation condition
#####################################

# parse worker indices from script arguments
idx_cond = 1

# STEP 1: Load pre-generated RNN parameters
###########################################

config = pickle.load(open("data/qif_rc_inh_config.pkl", 'rb'))

# connectivity matrix
C = config["C"]

# QIF input
inp = np.zeros_like(config['inp'])

# input weights
W_in = config['W_in']

# simulation config
dt = config['dt']
dts = config['dts']
cutoff = config['cutoff']
T = config['T']
t = int((T - cutoff)/dts)
M = config['number_input_channels']

# target values
y = config['targets']

# STEP 1: define remaining network parameters
#############################################

# general parameters
N = C.shape[0]
m = W_in.shape[0]
n_folds = 5
ridge_alpha = 0.5*10e-3

# qif parameters
Delta = 0.1
eta = 1.0
tau_a = 10.0
tau_s = 1.0

# adaptation strength
alpha = 0.2

# independent variable (IV)
iv_name = "J"
n_iv = 3
ivs = np.asarray([-12.5, -10.0, -7.5])  #np.linspace(8.1, 8.4, num=n_iv)

# mean-field parameters
C_m = np.ones(shape=(1,))

# STEP 3: Evaluate classification performance of RNN
####################################################

data = dict()
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

    # setup QIF RNN
    qif_rnn = QIFExpAddSyns(C, eta, iv, Delta=Delta, alpha=alpha, tau_s=tau_s, tau_a=tau_a, tau=1.0)

    # perform simulation
    results = qif_rnn.run(T, dt, dts, inp=inp, W_in=W_in, cutoff=cutoff, outputs=(np.arange(0, N), np.arange(3*N, 4*N)),
                          verbose=False)
    v_qif = np.mean(results[0], axis=1)
    r_qif = np.mean(results[1], axis=1)
    X = results[1]

    # simulate mean-field dynamics
    qif_mf = mQIFExpAddSynsRNN(C_m, eta, iv, Delta=Delta, alpha=alpha, tau_a=tau_a, tau_s=tau_s, tau=1.0)
    results = qif_mf.run(T, dt, dts, cutoff=cutoff, outputs=([0], [1]))
    v_mf = np.squeeze(results[0])
    r_mf = np.squeeze(results[1])

    # calculate Kuramoto order parameter Z for QIF network and mean-field model
    Z_qif = kuramoto_order_parameter(r_qif, v_qif)
    Z_mf = kuramoto_order_parameter(r_mf, v_mf)

    # store data
    data["r_qif"][j, :] = r_qif
    data["v_qif"][j, :] = v_qif
    data["r_mf"][j, :] = r_mf
    data["v_mf"][j, :] = v_mf
    data["Z_qif"][j, :] = Z_qif
    data["Z_mf"][j, :] = Z_mf

data["T"] = T
pickle.dump(data, open('data/qif_micro_macro_results.pkl', 'wb'))
