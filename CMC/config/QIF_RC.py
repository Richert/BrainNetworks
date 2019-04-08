import numpy as np
from scipy.spatial.distance import hamming


class aEIF_micro(object):

    def __init__(self, W, k=200.0, c=281.0, g_leak=30.0, v_leak=-70.6, delta_t=2.0, v_t=-50.4, v_theta=20.0,
                 v_r=-70.6, a=4.0, tau_x=144.0, delta_x=0.0805, tau_s=5.0, tau_r=900.0, tau_f=100.0, u_r=0.8,
                 tau_a=2.0, tau_q1=16.8, tau_q2=101.0, tau_o1=33.7, tau_o2=114.0, eta=1.0,
                 a_p2=0.0, a_p3=6.5e-3, a_d2=7.1e-3, a_d3=0.0, dt=0.01):

        self.N = W.shape[0]                     # number of EIF neurons
        self.W = W                              # coupling matrix
        self.k = k                              # global synaptic connectivity scaling
        self.c = c                              # membrane capacitance in pF
        self.g_leak = g_leak                    # maximum leak conductance in nS
        self.v_leak = v_leak                    # resting membrane potential in mV
        self.delta_t = delta_t                  # spike steepness in mV
        self.v_t = v_t                          # spike initiation threshold in mV
        self.v_theta = v_theta                  # spike emission threshold in mV
        self.v_r = v_r                          # reset membrane potential in mV
        self.a = a                              # peak conductance of adaptation current in nS
        self.tau_x = tau_x                      # time-scale of the adaptation current in ms
        self.delta_x = delta_x                  # increase in adaptation current after spike in nA
        self.tau_s = tau_s                      # excitatory synaptic time-constant in ms
        self.tau_r = tau_r                      # time constant of synaptic recovery after short-term depression in ms
        self.tau_f = tau_f                      # time constant of synaptic facilitation in ms
        self.u_r = u_r                          # pre-synaptic vesicle release probability
        self.dt = dt                            # euler integration time-step in ms
        self.tau_q1 = tau_q1                    # time-scale of firing rate running average q1
        self.tau_q2 = tau_q2                    # time-scale of firing rate running average q2
        self.tau_o1 = tau_o1                    # time-scale of firing rate running average o1
        self.tau_o2 = tau_o2                    # time-scale of firing rate running average o2
        self.eta = eta                          # global rate of STDP
        self.a_p2 = a_p2                        # LTP amplitude
        self.a_p3 = a_p3                        # LTP amplitude for triplet events
        self.a_d2 = a_d2                        # LTD amplitude
        self.a_d3 = a_d3                        # LTD amplitude for triplet events
        self.refractory_period = int(tau_a / self.dt)

        # state variables
        self.v = np.zeros((self.N, 1)) + self.v_r
        self.x = np.zeros_like(self.v)
        self.spike = np.zeros_like(self.v)
        self.wait = np.zeros_like(self.v)
        self.mask = np.ones((self.N, 1), dtype=bool)
        self.I_s = np.zeros_like(self.v)
        self.U = np.zeros_like(self.W) + self.u_r
        self.R = np.ones_like(self.W)
        self.q1 = np.zeros_like(self.v)
        self.q2 = np.zeros_like(self.v)
        self.o1 = np.zeros_like(self.v)
        self.o2 = np.zeros_like(self.v)

    def run(self, T, I=None, dts=None, **kwargs):

        if not dts:
            dts = self.dt
        store = int(dts/self.dt)
        steps_store = int(T/dts)
        steps_sim = int(T/self.dt)

        v_col = np.zeros((steps_store+1, self.N))
        spike_col = np.zeros((steps_sim, self.N), dtype=float)
        if I is None:
            I = np.zeros((steps_sim, self.N)).T
        elif I.shape[0] != self.N:
            I = I.T

        v_tmp = []
        s = 0
        for step in range(steps_sim):

            # calculate new spikes
            self.elicit_spikes()

            # project spikes through network and update synapses
            self.project()

            # update state variables
            self.state_update(I[:, step:step+1])

            # store variables
            spike_col[step, :] = self.spike
            v_tmp.append(self.v)
            if np.mod(step, store) == 0:
                v_col[s] = np.mean(v_tmp, axis=0)
                s += 1
                v_tmp.clear()

        return v_col, spike_col

    def state_update(self, inp_current):

        v_delta = (self.g_leak*(self.v_leak - self.v) +
                   self.g_leak*self.delta_t*np.exp((self.v - self.v_t)/self.delta_t) -
                   self.x + self.I_s + inp_current) / self.c
        x_delta = (self.a*(self.v_leak - self.v) - self.x) / self.tau_x

        self.v += self.dt * self.mask * v_delta
        self.x += self.dt * x_delta
        self.wait[self.wait > 0] -= 1

    def project(self, r=1.0):

        # spike-timing dependent updates
        if self.spike.any():
            spike = self.spike[:, 0]
            self.I_s += self.k*(self.W * self.U * self.R) @ self.spike
            self.W[:, spike] -= r * (self.eta * self.o1 * (self.a_d2 + self.a_d3 * (self.q2 - 1.)))
            self.W[spike, :] += r * (self.eta * self.q1 * (self.a_p2 + self.a_p3 * (self.o2 - 1.))).T
            self.W[self.W > 5.0] = 5.0
            self.R[:, spike] -= self.U[:, spike]*self.R[:, spike]
            self.U[:, spike] += self.u_r * (1.0 - self.U[:, spike])

        # decay term updates
        self.I_s += self.dt * (-self.I_s / self.tau_s)
        self.R += self.dt * ((1.0 - self.R) / self.tau_r)
        self.U += self.dt * ((self.u_r - self.U) / self.tau_f)
        self.q1 += self.dt * (-self.q1 / self.tau_q1)
        self.q2 += self.dt * (-self.q2 / self.tau_q2)
        self.o1 += self.dt * (-self.o1 / self.tau_o1)
        self.o2 += self.dt * (-self.o2 / self.tau_o2)

    def elicit_spikes(self):

        self.spike = self.v > self.v_theta
        if self.spike.any():
            self.wait[self.spike] = self.refractory_period
            self.v[self.spike] = self.v_r
            self.x[self.spike] += self.delta_x
            self.q1[self.spike] += 1.0
            self.q2[self.spike] += 1.0
            self.o1[self.spike] += 1.0
            self.o2[self.spike] += 1.0
        self.mask[self.wait != 0] = 0.
        self.mask[self.wait == 0.] = 1.


class IFRC(object):

    def __init__(self, W_intra, W_inter, W_out, W_fb, D_out, k, output_neurons, tau_r, dt):

        self.R = aEIF_micro(W_intra, dt=dt, tau_r=600.0, eta=8.0, k=k)
        self.F = aEIF_micro(W_inter, dt=dt, tau_r=200.0, tau_f=150.0, u_r=1.0, k=k, tau_s=7.0, eta=3.0)
        self.W_out = W_out
        self.W_fb = W_fb
        self.output_neurons = output_neurons
        self.dt = dt
        self.tau_r = tau_r
        self.k = k

        self.I_fr = np.zeros_like(self.F.v)
        self.I_rf = np.zeros_like(self.R.v)
        self.U_fr = np.zeros((self.F.N, self.R.N))
        self.U_rf = np.zeros((self.R.N, self.F.N))
        self.R_fr = np.zeros((self.F.N, self.R.N))
        self.R_rf = np.zeros((self.R.N, self.F.N))
        self.output = np.zeros((len(self.output_neurons), 1))

        dmax = int(np.max(D_out / dt)) + 2
        self.D_out = np.asarray(np.rint(D_out/dt), dtype=int)
        self.F_idx, self.R_idx = np.ogrid[0:D_out.shape[0], 0:D_out.shape[1]]
        self.spike_buffer = np.zeros((dmax, D_out.shape[0], D_out.shape[1]))
        self.reward = 0.
        self.max_error = self.calculate_error(np.ones_like(self.output), np.zeros_like(self.output))

    def run(self, T, target, I_r=None, I_f=None, dts=None, **kwargs):

        if not dts:
            dts = self.dt
        store = int(dts/self.dt)
        steps_store = int(T/dts)
        steps_sim = int(T/self.dt)
        N = len(self.output_neurons)

        v_col_r = np.zeros((steps_store+1, self.R.N))
        v_col_f = np.zeros((steps_store + 1, self.F.N))
        spike_col = np.zeros((steps_sim, N), dtype=float)
        if I_r is None:
            I_r = np.zeros((steps_sim, self.R.N))
        elif I_r.shape[0] != self.R.N:
            I_r = I_r.T
        if I_r is None:
            I_f = np.zeros((steps_sim, self.F.N))
        elif I_f.shape[0] != self.F.N:
            I_f = I_f.T

        v_tmp_r = []
        v_tmp_f = []
        s = 0
        for step in range(steps_sim):

            # calculate new spikes
            self.R.elicit_spikes()
            self.F.elicit_spikes()

            # project spikes through network and update synapses
            self.R.project(r=self.reward)
            self.F.project(r=self.reward)
            self.project()

            # update state variables
            self.R.state_update(I_r[:, step:step+1] + self.I_rf)
            self.F.state_update(I_f[:, step:step+1] + self.I_fr)

            # calculate reward
            self.calculate_reward(target[step:step+1])

            # store variables
            spike_col[step, :] = self.output[:, 0]
            v_tmp_r.append(self.R.v.squeeze())
            v_tmp_f.append(self.F.v.squeeze())
            if np.mod(step, store) == 0:
                v_col_r[s] = np.mean(v_tmp_r, axis=0)
                v_col_f[s] = np.mean(v_tmp_f, axis=0)
                s += 1
                v_tmp_r.clear()
                v_tmp_f.clear()

        return v_col_r[:-1], v_col_f[:-1], spike_col

    def project(self):

        # update spike buffer variables
        self.spike_buffer[:, :-1] = self.spike_buffer[:, 1:]
        self.spike_buffer[:, -1] = 0.
        if self.R.spike.any():
            self.spike_buffer[self.D_out[:, self.R.spike[:, 0]], self.F_idx, self.R_idx[None, self.R.spike.T]] = 1.0

        # project from reservoir R to feedback population F
        ###################################################

        # spike-timing dependent updates
        if self.R.spike.any():
            spike = self.R.spike[:, 0]
            q2 = (self.W_out > 0.) @ (self.R.q2 * self.R.spike)
            self.W_out[:, spike] -= self.reward * (self.F.eta * self.F.o1 * (self.F.a_d2 + self.F.a_d3 * (q2 - 1.)))
            self.R_fr[:, spike] -= self.U_fr[:, spike] * self.R_fr[:, spike]
            self.U_fr[:, spike] += self.F.u_r * (1.0 - self.U_fr[:, spike])
        if self.F.spike.any():
            spike = self.F.spike[:, 0]
            o2 = (self.W_fb > 0.) @ (self.F.o2 * self.F.spike)
            self.W_out[spike, :] += (self.reward * (self.F.eta * self.R.q1 * (self.F.a_p2 + self.F.a_p3 * (o2 - 1.)))).T
        if self.spike_buffer[0].any():
            self.I_fr += self.k * np.sum(self.W_out * self.U_fr * self.R_fr *
                                         self.spike_buffer[0], axis=1, keepdims=True)

        # decay term updates
        self.I_fr += self.dt * (-self.I_fr / self.F.tau_s)
        self.R_fr += self.dt * ((1.0 - self.R_fr) / self.F.tau_r)
        self.U_fr += self.dt * ((self.F.u_r - self.U_fr) / self.F.tau_f)

        # project from feedback population F to reservoir R
        ###################################################

        # spike-timing dependent updates
        if self.F.spike.any():
            spike = self.F.spike[:, 0]
            q2 = (self.W_fb > 0.) @ (self.F.q2 * self.F.spike)
            self.W_fb[:, spike] -= self.R.eta * self.R.o1 * (self.R.a_d2 + self.R.a_d3 * (q2 - 1.))
            self.R_rf[:, spike] -= self.U_rf[:, spike] * self.R_rf[:, spike]
            self.U_rf[:, spike] += self.R.u_r * (1.0 - self.U_rf[:, spike])
        if self.R.spike.any():
            spike = self.R.spike[:, 0]
            o2 = (self.W_out > 0.) @ (self.R.o2 * self.R.spike)
            self.W_fb[spike, :] += (self.R.eta * self.F.q1 * (self.R.a_p2 + self.R.a_p3 * (o2 - 1.))).T
        if self.F.spike.any():
            self.I_rf += self.k * (self.W_fb * self.U_rf * self.R_rf) @ self.F.spike

        # decay term updates
        self.I_rf += self.dt * (-self.I_rf / self.R.tau_s)
        self.R_rf += self.dt * ((1.0 - self.R_rf) / self.R.tau_r)
        self.U_rf += self.dt * (-self.U_rf / self.R.tau_f)

    def calculate_reward(self, target):

        self.output[:] = self.R.spike[self.output_neurons]
        error = self.calculate_error(self.output, target.T) if target.any() else 0.1
        reward_signal = 1. - error / self.max_error
        self.reward = self.dt * (-self.reward / self.tau_r) + reward_signal

    @staticmethod
    def calculate_error(x, y):
        return hamming(x, y)

    @staticmethod
    def delay_buffer_indices(D, dt):
        D = np.asarray(np.rint(D/dt), dtype=int)
        delay_indices = []
        for target in range(D.shape[0]):
            delay_indices.append([(target, D[target, source]) for source in range(D.shape[1])])
        return np.asarray(delay_indices, dtype=int)


class EIC(object):

    def __init__(self, W_ee, W_ii, W_ei, W_ie, k, dt=1e-3, e_kwargs={}, i_kwargs={}):

        self.E = aEIF_micro(W_ee, dt=dt, k=k, **e_kwargs)
        self.I = aEIF_micro(W_ii, dt=dt, k=k, **i_kwargs)
        self.W_ei = W_ei
        self.W_ie = W_ie
        self.dt = dt
        self.k = k

        self.I_ei = np.zeros_like(self.E.v)
        self.I_ie = np.zeros_like(self.I.v)
        self.U_ei = np.zeros((self.E.N, self.I.N))
        self.U_ie = np.zeros((self.I.N, self.E.N))
        self.R_ei = np.zeros((self.E.N, self.I.N))
        self.R_ie = np.zeros((self.I.N, self.E.N))

    def run(self, T, I_e=None, I_i=None, dts=None, **kwargs):

        if not dts:
            dts = self.dt
        store = int(dts/self.dt)
        steps_store = int(T/dts)
        steps_sim = int(T/self.dt)
        N = self.E.N

        v_col = np.zeros((steps_store+1, N))
        spike_col = np.zeros((steps_sim, N), dtype=float)
        if I_e is None:
            I_e = np.zeros((steps_sim, self.E.N))
        elif I_e.shape[0] != self.E.N:
            I_e = I_e.T
        if I_i is None:
            I_i = np.zeros((steps_sim, self.I.N))
        elif I_i.shape[0] != self.I.N:
            I_i = I_i.T

        v_tmp = []
        s = 0
        for step in range(steps_sim):

            # calculate new spikes
            self.E.elicit_spikes()
            self.I.elicit_spikes()

            # project spikes through network and update synapses
            self.E.project()
            self.I.project()
            self.project()

            # update state variables
            self.E.state_update(I_e[:, step:step+1] + self.I_ei)
            self.I.state_update(I_i[:, step:step+1] + self.I_ie)

            # store variables
            spike_col[step, :] = self.E.spike.squeeze()
            v_tmp.append(self.E.v.squeeze())
            if np.mod(step, store) == 0:
                v_col[s] = np.mean(v_tmp, axis=0)
                s += 1
                v_tmp.clear()

        return v_col[:-1], spike_col

    def project(self):

        # project from E to I population
        ################################

        # spike-timing dependent updates
        if self.E.spike.any():
            spike = self.E.spike[:, 0]
            q2 = (self.E.q2 * self.E.spike).T
            self.W_ie -= self.I.eta * self.I.o1 * (self.I.a_d2 + self.I.a_d3 * (q2 - 1.))
            self.R_ie[:, spike] -= self.U_ie[:, spike] * self.R_ie[:, spike]
            self.U_ie[:, spike] += self.I.u_r * (1.0 - self.U_ie[:, spike])
            self.I_ie += self.k * (self.W_ie * self.U_ie * self.R_ie) @ self.E.spike
        if self.I.spike.any():
            o2 = (self.I.o2 * self.I.spike).T
            self.W_ie += (self.I.eta * self.E.q1 * (self.I.a_p2 + self.I.a_p3 * (o2 - 1.))).T

        # decay term updates
        self.I_ie += self.dt * (-self.I_ie / self.I.tau_s)
        self.R_ie += self.dt * ((1.0 - self.R_ie) / self.I.tau_r)
        self.U_ie += self.dt * ((self.I.u_r - self.U_ie) / self.I.tau_f)

        # project from I to E population
        ################################

        # spike-timing dependent updates
        if self.I.spike.any():
            spike = self.I.spike[:, 0]
            q2 = (self.I.q2 * self.I.spike).T
            self.W_ei -= self.E.eta * self.E.o1 * (self.E.a_d2 + self.E.a_d3 * (q2 - 1.))
            self.R_ei[:, spike] -= self.U_ei[:, spike] * self.R_ei[:, spike]
            self.U_ei[:, spike] += self.E.u_r * (1.0 - self.U_ei[:, spike])
            self.I_ei += self.k * (self.W_ei * self.U_ei * self.R_ei) @ self.I.spike
        if self.E.spike.any():
            o2 = (self.E.o2 * self.E.spike).T
            self.W_ei += (self.E.eta * self.I.q1 * (self.E.a_p2 + self.E.a_p3 * (o2 - 1.))).T

        # decay term updates
        self.I_ei += self.dt * (-self.I_ei / self.E.tau_s)
        self.R_ei += self.dt * ((1.0 - self.R_ei) / self.E.tau_r)
        self.U_ei += self.dt * (-self.U_ei / self.E.tau_f)
