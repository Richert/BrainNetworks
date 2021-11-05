import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numba import njit
import pickle

# load results and config
##############

# load results
idx = 0
path = '/home/rgast/PycharmProjects/BrainNetworks/BasalGanglia/config'
params = np.load(f"{path}/stn_gpe_str_params_{idx}.npy")
loss = np.load(f"{path}/stn_gpe_str_loss_{idx}.npy")

# load config
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


# display fitted parameters
###########################

# examine optimized solution
############################

# extract excitabilities
eta_p, eta_a, eta_s, eta_d1, eta_d2, eta_f = params[:6]
print(f'STN excitability: {eta_s}')
print(f'GPe-p excitability: {eta_p}')
print(f'GPe-a excitability: {eta_a}')
print(f'MSN-D1 excitability: {eta_d1}')
print(f'MSN-D2 excitability: {eta_d2}')
print(f'FSI excitability: {eta_f}')

# extract connectivity parameters
k_pp, k_ap, k_sp, k_fp, k_d1a, k_d2a, k_ps, k_as, k_ad1, k_d1d1, k_pd2, k_d1d2, k_d2d2, k_d1f, k_d2f, k_ff = params[6:22]
print(f'GPe-p -> GPe-p: {k_pp}')
print(f'GPe-p -> GPe-a: {k_ap}')
print(f'GPe-p -> STN: {k_sp}')
print(f'GPe-a -> MSN-D1: {k_d1a}')
print(f'GPe-a -> MSN-D2: {k_d2a}')
print(f'STN -> GPe-p: {k_ps}')
print(f'STN -> GPe-a: {k_as}')
print(f'MSN-D1 -> GPe-a: {k_ad1}')
print(f'MSN-D1 -> MSN-D1: {k_d1d1}')
print(f'MSN-D2 -> GPe-p: {k_pd2}')
print(f'MSN-D2 -> MSN-D1: {k_d1d2}')
print(f'MSN-D2 -> MSN-D2: {k_d2d2}')
print(f'FSI -> MSN-D1: {k_d1f}')
print(f'FSI -> MSN-D2: {k_d2f}')
print(f'FSI -> FSI: {k_ff}')

# extract SFA parameters
tau_a, alpha_a, tau_d1, alpha_d1, tau_d2, alpha_d2 = params[22:28]
print(f'GPe-a adaptation time constant: {tau_a}')
print(f'GPe-a adaptation rate: {alpha_a}')
print(f'MSN-D1 adaptation time constant: {tau_d1}')
print(f'MSN-D1 adaptation rate: {alpha_d1}')
print(f'MSN-D2 adaptation time constant: {tau_d2}')
print(f'MSN-D2 adaptation rate: {alpha_d2}')

# extract other parameters
g_f, exc_d2, inh_stn, exc_stn = params[28:]
print(f'FSI gap junction strength: {g_f}')
print(f'Extrinsic stimulation strength of MSN-D2 (excitatory): {exc_d2}')
print(f'Extrinsic stimulation strength of STN (excitatory): {exc_stn}')
print(f'Extrinsic stimulation strength of STN (inhibitory): {inh_stn}')

# perform simulation
####################

# extract simulation parameters
args = config['args']
indices = config['indices']
conditions = config['conditions']
T = config['T']
dt = config['dt']
dts = config['dts']
targets = config['targets']

# overwrite default values in args
for p, idx in zip(params, indices):
    try:
        args[idx[0]][idx[1]] = p
    except IndexError:
        args[idx[0]] = p

# simulate system dynamics for different conditions
rates = run_all_conditions(conditions, rhs_func, args, params, state_vec, T, dt, dts,
                           method='LSODA', atol=1e-6, rtol=1e-5)

# plotting
##########

rate_indices = [1, 26, 0, 28]
fig, axes = plt.subplots(ncols=4, nrows=3, figsize=(12, 8))

for i in range(3):

    # GPe-p
    ax1 = axes[i, 0]
    ax1.plot(rates[i][:, rate_indices[0]], color='red')
    ax1.plot(targets[0][i], ls='--', color='black')
    ax1.set_title('GPe-p')

    # GPe-a
    ax2 = axes[i, 1]
    ax2.plot(rates[i][:, rate_indices[1]], color='red')
    ax2.plot(targets[1][i], ls='--', color='black')
    ax2.set_title('GPe-a')

    # STN
    ax3 = axes[i, 2]
    ax3.plot(rates[i][:, rate_indices[2]], color='red')
    ax3.plot(targets[2][i], ls='--', color='black')
    ax3.set_title('STN')

    # MSN-D2
    ax4 = axes[i, 3]
    ax4.plot(rates[i][:, rate_indices[3]], color='red')
    if targets[3][i] is not None:
        ax4.plot(targets[3][i], ls='--', color='black')
    ax4.set_title('MSN-D2')

plt.show()
