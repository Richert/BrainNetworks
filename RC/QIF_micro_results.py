import numpy as np
import pandas as pd
from pyrates.utility.visualization import plot_connectivity
from matplotlib.pyplot import show
import pickle, os

path = "/home/rgast/ownCloud/data/qif_rc_scores/lorenz_vs_stuartlandau"

# STEP 1: Load pre-generated RNN parameters
###########################################

config = pickle.load(open(f"{path}/qif_micro_config.pkl", 'rb'))

# connectivity matrix
C = config['C']

# input
inp = config['inp']

# input weights
W_in = config['W_in']

# simulation config
T = config['T']
dt = config['dt']
dts = config['dts']
cutoff = config['cutoff']

# target values
targets = config['targets']

# STEP 2: Go through results files and store scores
###################################################

etas, alphas, scores = [], [], []
for f in [f for f in os.listdir(path) if '.npy' in f]:
    idx = int(f.split(".")[0].split("_")[-1])
    alphas.append(config['alphas'][idx])
    etas.append(config['etas'][idx])
    scores.append(np.load(f"{path}/{f}"))

# STEP 3: Visualization
#######################

# visualization
etas_unique = np.sort(np.unique(etas))
alphas_unique = np.sort(np.unique(alphas))
scores_2d = np.zeros((len(alphas_unique), len(etas_unique)))
for eta, alpha, score in zip(etas, alphas, scores):
    idx_c = np.argwhere(etas_unique == eta)
    idx_r = np.argwhere(alphas_unique == alpha)
    if score >= 0.0:
        scores_2d[idx_r, idx_c] = score

plot_connectivity(scores_2d, xticklabels=np.round(etas_unique, decimals=2),
                  yticklabels=np.round(alphas_unique, decimals=2))
show()
