import numpy as np


# fitness functions
def fitness(y, t):
    y = np.asarray(y).flatten()
    t = np.asarray(t).flatten()
    diff = np.asarray([0.0 if np.isnan(t_tmp) else y_tmp - t_tmp for y_tmp, t_tmp in zip(y, t)]).flatten()
    t[np.isnan(t)] = 1.0
    t[t == 0] = 1.0
    weights = 1/np.abs(t)
    return weights @ np.abs(diff)


def spectral_fitness(freqs, freq_targets, vars):
    y, targets = [], []
    for i in range(len(freqs)):
        if np.isnan(freq_targets[i]):
            y.append(0.0)
        elif freq_targets[i] == 0.0:
            y.append(vars[i])
        else:
            y.append(freqs[i])
        targets.append(freq_targets[i])
    return fitness(y, targets)


# data
targets = [[20, 60],   # healthy control
           [np.nan, 40],  # ampa blockade in GPe
           [np.nan, 60],  # ampa and gabaa blockade in GPe
           [np.nan, 100],  # GABAA blockade in GPe
           [np.nan, 30],  # STN blockade
           [40, 100],  # GABAA blockade in STN
           [30, 40],  # MPTP-induced PD
           [np.nan, 20],  # PD + ampa blockade in GPe
           [np.nan, 90],  # PD + gabaa blockade in GPe
           [30, np.nan],  # PD + AMPA blockade in STN
           [40, np.nan],  # PD + GPe blockade
           ]
freq_targets = [0.0, np.nan, np.nan, np.nan, np.nan, np.nan, 14.0, 0.0, 14.0, np.nan, np.nan]

fr_dev = 10.0
freq_dev = 1.0
var_dev = 0.5

outputs, variances, freqs = [], [], []
for target, freq in zip(targets, freq_targets):
    r_e = target[0] + fr_dev
    r_i = target[1] + fr_dev
    outputs.append([r_e, r_i])
    variances.append(var_dev)
    freqs.append(freq + freq_dev)

dist1 = fitness(outputs, targets)
dist2 = spectral_fitness(freqs, freq_targets, variances)
print(1/(dist1+dist2))
