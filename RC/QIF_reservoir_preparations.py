import numpy as np
from scipy.integrate import solve_ivp
import pickle

# simulation parameters
T = 130.0
dt = 1e-4
dts = 1e-2
cutoff = 10.0

# network configuration parameters
N = 2000
p = 0.05
m = 1
n_etas = 100
n_alphas = 10
etas = np.linspace(-6.5, -4.5, num=n_etas)
alphas = np.linspace(0.0, 0.1, num=n_alphas)

# input parameters
a = 10.0
b = 28.0
c = 8/3
omega = 1.25664

# setup connectivity matrix
C = np.random.rand(N, N)
C_sorted = np.sort(C.flatten())
C = 1. * (C <= C_sorted[int(N**2*p)])
C /= N*p

# setup input matrix
W_in = np.random.rand(N, m)
W_sorted = np.sort(W_in.flatten())
W_in[np.abs(W_in) < W_sorted[int(N*m*0.5)]] = 0.0

# create lorenz input
y_delta = np.zeros((3,))
def lorenz(t, y):
    y_delta[0] = a*(y[1] - y[0])
    y_delta[1] = y[0]*(b - y[2]) - y[1]
    y_delta[2] = y[0]*y[1] - c*y[2]
    return y_delta

sim_time = (T-cutoff)*0.5
t_eval = np.arange(0.0, sim_time, dt)
y_lorenz = solve_ivp(lorenz, t_span=[0.0, sim_time], y0=[1.0, 0.0, 0.0], t_eval=t_eval).y

# create stuart-landau input
y_delta = np.zeros((2,))
def stula(t, y):
    y_delta[0] = -omega*y[1] + y[0]*(1 - y[0]**2 - y[1]**2)
    y_delta[1] = omega*y[0] + y[1]*(1 - y[0]**2 - y[1]**2)
    return y_delta
y_stula = solve_ivp(stula, t_span=[0.0, sim_time], y0=[1.0, 0.0,], t_eval=t_eval).y

# create input and target matrix
steps = int(np.round(T/dt))
store_steps = int(np.round((T-cutoff)/dts))
in_start = int(np.round(cutoff/dt))
in_end = int(np.round(sim_time/dt))
target_start = int(np.round(0.0/dts))
target_end = int(np.round(sim_time/dts))

inp = np.zeros((1, steps))
inp[0, in_start:in_start+in_end] = y_lorenz[0, :]
inp[0, in_start+in_end:] = y_stula[0, :]
targets = np.zeros((store_steps,))
targets[target_start:target_start+target_end] = 1.0
targets[target_start+target_end:] = -1.0

# store data
data = {}
data['T'] = T
data['dt'] = dt
data['dts'] = dts
data['cutoff'] = cutoff
data['N'] = N
data['p'] = p
data['C'] = C
data['W_in'] = W_in
data['inp'] = inp
data['targets'] = targets
data['etas'] = etas
data['alphas'] = alphas
fn = "/u/rgast/ptmp_link/BrainNetworks/RC/results/qif_micro_config.pkl"
pickle.dump(data, open(fn, 'wb'))
