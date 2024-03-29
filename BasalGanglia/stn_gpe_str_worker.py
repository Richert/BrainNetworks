import pickle
import sys
from scipy.optimize import differential_evolution
from scipy.integrate import solve_ivp
from numba import njit
import numpy as np
import time

# preparations and function definitions
#######################################

# parse worker indices from script arguments
idx_cond = 0 # int(sys.argv[1])

# load config
path = '/home/rgast/PycharmProjects/BrainNetworks/BasalGanglia/config'
config = pickle.load(open(f"{path}/stn_gpe_str_config.pkl", 'rb'))

# extract constants from config
(c, tau, tau_0, tau_1, delta, delta_0, delta_1, k_d1, k_d2, k_d3, k_d4, k_d5, k_d1_0, k_d2_0, k_d3_0,
 k_d4_0, k_d5_0, k_d1_1, k_d2_1, k_d3_1, k_d4_1, R_buffered_idx, R_buffered_idx_0, source_idx, source_idx_0,
 source_idx_1, source_idx_2, source_idx_3, source_idx_4, source_idx_5, source_idx_6, source_idx_7,
 source_idx_8, source_idx_9, target_idx, target_idx_0, target_idx_1, target_idx_2, target_idx_3,
 target_idx_4, target_idx_5) = config['constants']

# extract state-vector from config
state_vec = config['u_init']

# vector-field update function
@njit
def rhs_func(t, state_vec, state_vec_update,
             eta, eta_0, eta_1,
             weight, weight_0, weight_1, weight_2, weight_3,
             weight_4, weight_5, weight_6, weight_7,
             alpha, tau_a, g, w_in, w_in_0,
             R_buffered, R_buffered_0,
             s_e, s_i_0, s_e_0, s_i_1, inp, time
             ):

    # global variable definition
    pi = np.pi
    t_inp = np.interp(t, time, inp)

    # extract state variables of QIF populations
    ############################################

    # base QIF cells (STN, GPe-p)
    R = state_vec[0:2]
    V = state_vec[2:4]
    R_d1 = state_vec[4:10]
    R_d2 = state_vec[10:16]
    R_d3 = state_vec[16:22]
    R_d4 = state_vec[22:24]
    R_d5 = state_vec[24:26]

    # QIF cells with SFA (GPe-a, MSN-D1, MSN-D2)
    R_0 = state_vec[26:29]
    V_0 = state_vec[29:32]
    A_0 = state_vec[32:35]
    R_d1_0 = state_vec[35:42]
    R_d2_0 = state_vec[42:49]
    R_d3_0 = state_vec[49:56]
    R_d4_0 = state_vec[56:61]
    R_d5_0 = state_vec[61:63]

    # QIF cells with gap junctions (FSI)
    R_1 = state_vec[63]
    V_1 = state_vec[64]
    R_d1_1 = state_vec[65:68]
    R_d2_1 = state_vec[68:71]
    R_d3_1 = state_vec[71:74]
    R_d4_1 = state_vec[74:77]

    # update output buffers
    #######################

    # output of base QIF cells (STN, GPe-p)
    R_buffered[0:4] = R_d3[0:4]
    R_buffered[4:6] = R_d5[0:2]
    R_buffered_sorted = R_buffered[R_buffered_idx]

    # output of QIF cells with SFA (GPe-p, MSN-D1, MSN-D2)
    R_buffered_0[0:2] = R_d3_0[0:2]
    R_buffered_0[2:5] = R_d4_0[0:3]
    R_buffered_0[5:7] = R_d5_0[0:2]
    R_buffered_0_sorted = R_buffered_0[R_buffered_idx_0]

    # output of QIF cells with gap junctions (FSI)
    R_buffered_1 = R_d4_1

    # calculate synaptic inputs
    ###########################

    # inputs to base QIF cells
    s_i = np.dot(target_idx_0, weight_0 * R_buffered_sorted[source_idx_0])
    s_i_0[target_idx_1] = weight_1 * R_buffered_0_sorted[source_idx_1]
    s_e[target_idx] = weight * R_buffered_sorted[source_idx]
    s = s_e - s_i - s_i_0 + w_in * t_inp

    # inputs to QIF cells with SFA (GPe-a, MSN-D1, MSN-D2)
    s_i_1[target_idx_3] = weight_3 * R_buffered_sorted[source_idx_4]
    s_i_2 = np.dot(target_idx_4, weight_4 * R_buffered_0_sorted[source_idx_5])
    s_i_3 = np.dot(target_idx_5, weight_5 * R_buffered_1[source_idx_6])
    s_e_0[target_idx_2] = weight_2 * R_buffered_sorted[source_idx_3]
    s_0 = s_e_0 - s_i_1 - s_i_2 - s_i_3 + w_in_0 * t_inp

    # inputs to QIF cells with gap junctions (FSI)
    s_i_4 = weight_6 * R_buffered_sorted[source_idx_8]
    s_i_5 = weight_7 * R_buffered_1[source_idx_9]
    s_1 = - s_i_4 - s_i_5

    # update vector field
    #####################

    # base QIF cells (STN, GPe-p)
    state_vec_update[0:2] = (2.0 * R * V + delta / (pi * tau)) / tau
    state_vec_update[2:4] = (V ** 2 + (eta + s * tau) / c - (pi * R * tau) ** 2) / tau
    state_vec_update[4:10] = k_d1 * (R[source_idx_2] - R_d1)
    state_vec_update[10:16] = k_d2 * (R_d1 - R_d2)
    state_vec_update[16:22] = k_d3 * (R_d2 - R_d3)
    state_vec_update[22:24] = k_d4 * (R_d3[4:6] - R_d4)
    state_vec_update[24:26] = k_d5 * (R_d4 - R_d5)

    # QIF cells with SFA (GPe-a, MSN-D1, MSN-D2)
    state_vec_update[26:29] = (2.0 * R_0 * V_0 + delta_0 / (pi * tau_0)) / tau_0
    state_vec_update[29:32] = (V_0 ** 2 + (eta_0 + s_0 * tau_0 - A_0) / c - (pi * R_0 * tau_0) ** 2) / tau_0
    state_vec_update[32:35] = R_0 * alpha - A_0 / tau_a
    state_vec_update[35:42] = k_d1_0 * (R_0[source_idx_7] - R_d1_0)
    state_vec_update[42:49] = k_d2_0 * (R_d1_0 - R_d2_0)
    state_vec_update[49:56] = k_d3_0 * (R_d2_0 - R_d3_0)
    state_vec_update[56:61] = k_d4_0 * (R_d3_0[2:7] - R_d4_0)
    state_vec_update[61:63] = k_d5_0 * (R_d4_0[3:5] - R_d5_0)

    # QIF cells with gap junctions (FSI)
    state_vec_update[63] = ((R_1 * (2 * V_1 - g) + delta_1 / (pi * tau_1)) / tau_1)[0]
    state_vec_update[64] = ((V_1 ** 2 + (eta_1 + s_1 * tau_1) / c - (pi * R_1 * tau_1) ** 2) / tau_1)[0]
    state_vec_update[65:68] = k_d1_1 * (np.asarray([R_1, R_1, R_1]) - R_d1_1)
    state_vec_update[68:71] = k_d2_1 * (R_d1_1 - R_d2_1)
    state_vec_update[71:74] = k_d3_1 * (R_d2_1 - R_d3_1)
    state_vec_update[74:77] = k_d4_1 * (R_d3_1 - R_d4_1)

    return state_vec_update


# envelope extraction
def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min
    #lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1

    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s)
        # pre-sorting of locals min based on relative position with respect to s_mid
        #lmin = lmin[s[lmin] < s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid
        lmax = lmax[s[lmax] > s_mid]

    # global max of dmax-chunks of locals max
    #lmin = lmin[[i + np.argmin(s[lmin[i:i + dmin]]) for i in range(0, len(lmin), dmin)]]
    # global min of dmin-chunks of locals min
    lmax = lmax[[i + np.argmax(s[lmax[i:i + dmax]]) for i in range(0, len(lmax), dmax)]]

    return lmax


# root-mean-squared error function
@njit
def rmse(y, target):
    diff = (y - target) ** 2
    target[np.abs(target) < 1.0] = 1.0
    return np.sqrt(np.mean(diff / np.abs(target)))


# function that performs simulation of system dynamics for all conditions
def run_all_conditions(conditions, func, args, params, y, T, dt, dts, **kwargs):
    eval_times = np.linspace(0, T, num=int(np.round(T / dts)))
    rates = []
    for cond_input in conditions:
        # extract indices for which system parameters should be adjusted
        inp_idx, param_idx = cond_input

        # adjust input parameter
        old_inp = args[inp_idx[0]][inp_idx[1]]
        args[inp_idx[0]][inp_idx[1]] = params[param_idx]

        # perform simulation
        results = solve_ivp(fun=func, t_span=(0.0, T), y0=y, first_step=dt,
                            args=tuple(args), t_eval=eval_times, **kwargs)
        rates.append(results['y'].T * 1e3)

        # restore input parameter
        args[inp_idx[0]][inp_idx[1]] = old_inp

    return rates


def loss(params, indices, func, args, conditions, targets, T, dt, dts, kwargs):

    # overwrite default values in args
    for p, idx in zip(params, indices):
        try:
            args[idx[0]][idx[1]] = p
        except IndexError:
            args[idx[0]] = p

    # simulate system dynamics for different conditions
    results = run_all_conditions(conditions, func, args, params, state_vec, T, dt, dts,
                                 **kwargs)

    # calculate loss
    loss = 0
    rate_indices = [1, 26, 0, 28]
    for i, rates in enumerate(results):
        for j, target in enumerate(targets):
            t = target[i]
            if t is not None:
                y = rates[:, rate_indices[j]]
                # idx = hl_envelopes_idx(rates[:, rate_indices[j]])
                # loss += rmse(y[idx], t[idx])
                loss += rmse(y, t)

    return loss


# perform evolutionary optimization
###################################

# extract configuration constants
bounds = config['bounds']
indices = config['indices']
args = config['args']
conditions = config['conditions']
targets = config['targets']
T = config['T']
dt = config['dt']
dts = config['dts']

# start evolutionary optimization
de_iter = 2
de_size = 1
t0 = time.perf_counter()
results = differential_evolution(loss, bounds=bounds, disp=True, maxiter=de_iter, workers=1, popsize=de_size,
                                 polish=False, args=(indices, rhs_func, args, conditions, targets, T,  dt, dts,
                                                     {'method': 'LSODA', 'atol': 1e-6, 'rtol': 1e-5}))
t1 = time.perf_counter()
print(f'Time required for differential evolution optimization with {de_iter} iterations and a population size '
      f'of {int(de_size*32)}: {t1-t0} s.')

# save results to file
np.save(f"{path}/stn_gpe_str_params_{idx_cond}", results.x)
np.save(f"{path}/stn_gpe_str_loss_{idx_cond}", results.fun)
