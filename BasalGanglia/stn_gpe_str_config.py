import numpy as np
import pickle

################################
# generate target firing rates #
################################


# general parameters and functions
##################################


path = '/home/rgast/PycharmProjects/BrainNetworks/BasalGanglia/config'

# function definition for exponential response adaptation
def exp_adapt(t, u, eta, tau, inp):
    return inp(t) + (eta - u)/tau


# response collectors
stn_responses = []
gpe_p_responses = []
gpe_a_responses = []
msn_d2_responses = []

# simulation parameters
T = 6000.0
dt = 1.0
times = np.linspace(0, T, num=int(np.round(T/dt)))

# input parameters
w_in = np.asarray([3.0, 0.0])
w_in_0 = np.asarray([0.0, 0.0, 0.0])
start = int(np.round(2000.0/dt))
dur = int(np.round(2000.0/dt))
inp = np.zeros_like(times)
inp[start:start+dur] = 1.0
inp_time = np.linspace(0, 2000.0, num=dur)

# C1: excitation of MSN-D2 cells
################################

# msn-d2 response
eta_d2 = 0.0
tau_d2 = 600
k_d2 = 20.0
alpha = 50
msn_d2_response = eta_d2 + k_d2*inp
msn_d2_response[inp > 0] += alpha*np.exp(-inp_time/tau_d2)

# gpe-p response
eta_p = 25.0
tau_p = 300
k_p = -25.0
alpha = 2.0
gpe_p_response = eta_p + k_p*inp
gpe_p_response[start:start+dur] += alpha*np.exp(-inp_time[::-1]/tau_p)

# gpe-a response
eta_a = 5.0
tau_a = 100
k_a = 16.0
alpha = 1.0
inp_a = k_a*inp
inp_a[start:start+dur] *= alpha*(1-np.exp(-inp_time/tau_a))
gpe_a_response = eta_a + inp_a

# stn response
eta_s = 6.0
k_s = 12.0
tau_s = 800
alpha = 4.0
stn_response = eta_s + k_s*inp
stn_response[start:start+dur] += alpha*np.exp(-inp_time/tau_s)

# collect respones
gpe_p_responses.append(gpe_p_response)
gpe_a_responses.append(gpe_a_response)
stn_responses.append(stn_response)
msn_d2_responses.append(msn_d2_response)

# inhibition of STN cells
##########################

# stn response
eta_s = 6.0
tau_s = 200
k_s = -5.2
alpha = 7.0
stn_response = eta_s + k_s*inp
stn_response[start+dur:] += alpha*np.exp(-inp_time/tau_s)

# prototypical response
eta_p = 25.0
tau_p = 200
k_p = -15.0
alpha = -3.0
gpe_p_response = eta_p + k_p*inp
gpe_p_response[start:start+dur] += alpha*np.exp(-inp_time/tau_p)

# arkypallidal response
eta_a = 5.0
tau_a = 400
k_a = 8.0
alpha = 2.0
gpe_a_response = eta_a + k_a*inp
gpe_a_response[start:start+dur] += alpha*np.exp(-inp_time/tau_a)

# collect respones
gpe_p_responses.append(gpe_p_response)
gpe_a_responses.append(gpe_a_response)
stn_responses.append(stn_response)
msn_d2_responses.append(None)

# excitation of STN cells
##########################

# stn response
eta_s = 6.0
tau_s = 200
k_s = 16.0
alpha = 14.0
stn_response = eta_s + k_s*inp
stn_response[start+dur:] += alpha*np.exp(-inp_time/tau_s)
stn_response[start+dur:] -= 1.0

# prototypical response
eta_p = 25.0
tau_p = 200
k_p = 50.0
alpha = 10.0
gpe_p_response = eta_p + k_p*inp
gpe_p_response[start:start+dur] += alpha*np.exp(-inp_time/tau_p)
gpe_p_response[start+dur:] += 4.0

# arkypallidal response
eta_a = 5.0
tau_a = 200
k_a = -4.5
alpha = -0.5
gpe_a_response = eta_a + k_a*inp
gpe_a_response[start:start+dur] += alpha*np.exp(-inp_time/tau_a)
gpe_a_response[start+dur:] -= 0.8

# collect respones
gpe_p_responses.append(gpe_p_response)
gpe_a_responses.append(gpe_a_response)
stn_responses.append(stn_response)
msn_d2_responses.append(None)

#####################
# system parameters #
#####################

# constants
###########

c = 1/np.pi**2

tau = np.asarray([12.2, 14.7]) / c
tau_0 = np.asarray([23.4, 77.3, 60.2]) / c
tau_1 = np.asarray([17.2]) / c

delta = np.asarray([0.8, 6.5]) / c
delta_0 = np.asarray([2.6, 6.9, 2.4]) / c
delta_1 = np.asarray([1.5]) / c

k_d1 = np.asarray([0.75, 0.75, 1.0, 1.0, 5/3, 5/3])
k_d2 = k_d1
k_d3 = k_d1
k_d4 = k_d1[4:]
k_d5 = k_d4

k_d1_0 = np.asarray([1.0, 1.0, 2.0, 2.0, 2.0, 0.72463768, 0.72463768])
k_d2_0 = k_d1_0
k_d3_0 = k_d1_0
k_d4_0 = k_d1_0[2:]
k_d5_0 = k_d4_0[3:]

k_d1_1 = np.asarray([2.0, 2.0, 2.0])
k_d2_1 = k_d1_1
k_d3_1 = k_d1_1
k_d4_1 = k_d1_1

# helper variable initialization
################################

R_buffered_idx = np.asarray([0, 4, 1, 2, 5, 3])
R_buffered_idx_0 = np.asarray([0, 1, 2, 5, 3, 4, 6])

source_idx = 2
source_idx_0 = np.asarray([3, 4])
source_idx_1 = 6
source_idx_2 = np.asarray([0, 0, 1, 1, 1, 1])
source_idx_3 = 0
source_idx_4 = 1
source_idx_5 = np.asarray([0, 1, 2, 3, 4, 5])
source_idx_6 = np.asarray([0, 1])
source_idx_7 = np.asarray([0, 0, 1, 2, 2, 1, 2])
source_idx_8 = 5
source_idx_9 = 2

target_idx = np.asarray([1])
target_idx_0 = np.asarray([[1., 0.], [0., 1.]])
target_idx_1 = np.asarray([1])
target_idx_2 = np.asarray([0])
target_idx_3 = np.asarray([0])
target_idx_4 = np.asarray([[0., 0., 0., 1., 0., 0.],
                           [1., 0., 1., 0., 1., 0.],
                           [0., 1., 0., 0., 0., 1.]])
target_idx_5 = np.asarray([[0., 0.], [1., 0.], [0., 1.]])

constants = (c, tau, tau_0, tau_1, delta, delta_0, delta_1, k_d1, k_d2, k_d3, k_d4, k_d5, k_d1_0, k_d2_0, k_d3_0,
             k_d4_0, k_d5_0, k_d1_1, k_d2_1, k_d3_1, k_d4_1, R_buffered_idx, R_buffered_idx_0, source_idx, source_idx_0,
             source_idx_1, source_idx_2, source_idx_3, source_idx_4, source_idx_5, source_idx_6, source_idx_7,
             source_idx_8, source_idx_9, target_idx, target_idx_0, target_idx_1, target_idx_2, target_idx_3,
             target_idx_4, target_idx_5)

# Parameters to be optimized
############################

# background excitabilities
eta_p = -8.0
eta_a = 4.0
eta_s = 2.0
eta_d1 = 1.0
eta_d2 = 1.0
eta_f = 2.0

# GPe-p projection strengths
k = 40.0
k_pp = 2.0
k_ap = 3.0
k_sp = 2.0
k_fp = 1.0

# GPe-a projection strengths
k_d1a = 2.0
k_d2a = 2.0

# STN projection strengths
k_ps = 6.0
k_as = 2.0

# MSN-D1 projection strengths
k_ad1 = 2.0
k_d1d1 = 1.0

# MSN-D2 projection strengths
k_pd2 = 10.0
k_d1d2 = 5.0
k_d2d2 = 1.0

# FSI projection strengths
k_d1f = 5.0
k_d2f = 2.0
k_ff = 1.0

# SFA parameters
tau_a = 100.0
alpha_a = 1.0
tau_d1 = 200.0
alpha_d1 = 2.0
tau_d2 = 200.0
alpha_d2 = 2.0

# gap junction parameters
g_f = 0.4

# combine parameters
####################

eta = np.asarray([eta_s, eta_p])
eta_0 = np.asarray([eta_a, eta_d1, eta_d2])
eta_1 = np.asarray([eta_f])

weight = np.asarray([k_ps]) * k
weight_0 = np.asarray([k_sp, k_pp]) * k
weight_1 = np.asarray([k_pd2]) * k
weight_2 = np.asarray([k_as]) * k
weight_3 = np.asarray([k_ap]) * k
weight_4 = np.asarray([k_d1a, k_d2a, k_d1d1, k_ad1, k_d1d2, k_d2d2]) * k
weight_5 = np.asarray([k_d1f, k_d2f]) * k
weight_6 = np.asarray([k_fp]) * k
weight_7 = np.asarray([k_ff]) * k

R_buffered = np.zeros((6,))
R_buffered_0 = np.zeros((7,))

s_e = np.zeros((2,))
s_i_0 = np.zeros_like(s_e)
s_e_0 = np.zeros((3,))
s_i_1 = np.zeros_like(s_e_0)

taus = np.asarray([tau_a, tau_d1, tau_d2])
alphas = np.asarray([alpha_a, alpha_d1, alpha_d2])

state_vec = np.zeros((77,))
state_vec_update = np.zeros_like(state_vec)

# re-define time-dependent signals (more fine-grained)
T = 6000.0
dt = 1e-3
dts = 1.0
times = np.linspace(0, T, num=int(np.round(T/dt)))
start = int(np.round(2000.0/dt))
dur = int(np.round(2000.0/dt))
inp = np.zeros_like(times)
inp[start:start+dur] = 1.0

# final system parameters
system_args = [state_vec_update, eta, eta_0, eta_1, weight, weight_0, weight_1, weight_2,
               weight_3, weight_4, weight_5, weight_6, weight_7, alphas, taus, g_f, w_in, w_in_0,
               R_buffered, R_buffered_0, s_e, s_i_0, s_e_0, s_i_1, inp, times
               ]

# Evolutionary fitting parameter boundaries
###########################################

# define parameter boundaries
eta_p = (-10.0, 10.0)
eta_a = (-3.0, 12.0)
eta_s = (0.0, 15.0)
eta_d1 = (-10.0, 10.0)
eta_d2 = (-10.0, 10.0)
eta_f = (-5.0, 10.0)
k_pp = (0.0, 5.0)
k_ap = (0.0, 10.0)
k_sp = (0.0, 20.0)
k_fp = (0.0, 10.0)
k_d1a = (0.0, 50.0)
k_d2a = (0.0, 50.0)
k_ps = (0.0, 50.0)
k_as = (0.0, 20.0)
k_ad1 = (0.0, 100.0)
k_d1d1 = (0.0, 10.0)
k_pd2 = (0.0, 200.0)
k_d1d2 = (0.0, 20.0)
k_d2d2 = (0.0, 10.0)
k_d1f = (0.0, 100.0)
k_d2f = (0.0, 100.0)
k_ff = (0.0, 10.0)
tau_a = (50.0, 300.0)
alpha_a = (0.0, 5.0)
tau_d1 = (50.0, 300.0)
alpha_d1 = (0.0, 5.0)
tau_d2 = (50.0, 300.0)
alpha_d2 = (0.0, 5.0)
g_f = (0.0, 2.0)
exc_d2 = (0.01, 0.5)
exc_stn = (0.01, 0.5)
inh_stn = (-1.0, -0.1)

# final genes and gene indices
boundaries = [eta_p, eta_a, eta_s, eta_d1, eta_d2, eta_f, k_pp, k_ap, k_sp, k_fp,
              k_d1a, k_d2a, k_ps, k_as, k_ad1, k_d1d1,
              k_pd2, k_d1d2, k_d2d2, k_d1f, k_d2f, k_ff,
              tau_a, alpha_a, tau_d1, alpha_d1, tau_d2, alpha_d2, g_f,
              exc_d2, inh_stn, exc_stn
             ]
param_indices = [(1, 1), (2, 0), (1, 0), (2, 1), (2, 2), (3, 0), (5, 1), (8, 0), (5, 0), (11, 0),
                 (9, 0), (9, 1), (4, 0), (7, 0), (9, 3), (9, 2),
                 (6, 0), (9, 4), (9, 5), (10, 0), (10, 1), (12, 0),
                 (14, 0), (13, 0), (14, 1), (13, 1), (14, 2), (13, 2), (15,)]

# input conditions
conditions = [((17, 2), -3),  # MSN-D2 excitation
              ((16, 0), -2),  # STN inhibition
              ((16, 0), -1),  # STN excitation
             ]

# save config to file
#####################

data = dict()

data['T'] = T
data['dt'] = dt
data['dts'] = dts
data['targets'] = [stn_responses, gpe_p_responses, gpe_a_responses, msn_d2_responses]
data['args'] = system_args
data['bounds'] = boundaries
data['indices'] = param_indices
data['conditions'] = conditions
data['constants'] = constants
data['u_init'] = state_vec

pickle.dump(data, open(f"{path}/stn_gpe_str_config.pkl", 'wb'))
