import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import eigs
import pickle

path = "C:\\Users\\rgf3807\\PycharmProjects\\BrainNetworks}\\RC\\results"

# simulation parameters
T = 250.0
dt = 1e-3
dts = 1e-3
cutoff = 50.0

# network configuration parameters
N = 800
p = 0.2
m = 2
n_etas = 20
n_alphas = 10
etas = np.linspace(-2.0, 0.0, num=n_etas)
alphas = np.linspace(0.0, 0.3, num=n_alphas)
etas_all = np.tile(etas, reps=(n_alphas,))
alphas_all = []
for i in range(n_alphas):
    alphas_all += [alphas[i]]*n_etas
alphas_all = np.asarray(alphas_all)

# input parameters
a = 10.0
b = 28.0
c = 8/3
omega = 2*np.pi/5

# setup connectivity matrix
neurons = np.arange(0, N)
C = np.random.uniform(low=1e-4, high=1, size=(N, N))
n_incoming = int(N*(1-p))
for i in range(N):
    C[np.random.choice(neurons, size=n_incoming, replace=False), i] = 0
vals, vecs = eigs(C, k=int(N/10))
sr = np.max(np.real(vals))
C /= sr

# setup input matrix
p_in = 0.05
W_in = np.random.randn(N, m)
W_sorted = np.sort(W_in.flatten())
idx = np.abs(W_in) < W_sorted[int(N*m*p_in)]
W_in[idx] = 0.0
idx2 = np.abs(W_in) >= W_sorted[int(N*m*p_in)]
W_in[idx2] = np.random.uniform(-1, 1, np.sum(idx2))

# create lorenz input
y_delta = np.zeros((3,))
def lorenz(t, y):
    y_delta[0] = a*(y[1] - y[0])
    y_delta[1] = y[0]*(b - y[2]) - y[1]
    y_delta[2] = y[0]*y[1] - c*y[2]
    return y_delta


n_epochs = 20
sim_time = T*0.5/n_epochs
t_eval = np.arange(0.0, sim_time, dt)
y_lorenz = solve_ivp(lorenz, t_span=[0.0, sim_time], y0=[1.0, 0.0, 0.0], t_eval=t_eval, method='DOP853').y

# create stuart-landau input
y_delta = np.zeros((2,))
def stula(t, y):
    y_delta[0] = -omega*y[1] + y[0]*(1 - y[0]**2 - y[1]**2)
    y_delta[1] = omega*y[0] + y[1]*(1 - y[0]**2 - y[1]**2)
    return y_delta
y_stula = solve_ivp(stula, t_span=[0.0, sim_time], y0=[1.0, 0.0,], t_eval=t_eval, method='DOP853').y


# create input and target matrix
steps = int(np.round(T/dt))
store_steps = int(np.round((T-cutoff)/dts))
in_dur = int(np.round(sim_time/dt))
in_starts = np.arange(0, steps, 2*in_dur)
target_length = int(np.round(sim_time/dts))

in_scale = 2.0
inp = np.zeros((m, steps))
targets = np.zeros((store_steps,))
i = 0
for in_start in in_starts:
    inp[0, in_start:in_start+in_dur] = y_lorenz[0, :] * in_scale/10
    inp[1, in_start+in_dur:in_start+2*in_dur] = y_stula[0, :] * in_scale
    if in_start*dt >= cutoff:
        targets[i*target_length:(i+1)*target_length] = 1.0
        targets[(i+1)*target_length:(i+2)*target_length] = -1.0
        i += 2

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
data['etas'] = etas_all
data['alphas'] = alphas_all
pickle.dump(data, open(f"{path}\\qif_micro_config.pkl", 'wb'))
