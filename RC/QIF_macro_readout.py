import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedKFold
from pyrates.utility.visualization import plot_connectivity
from matplotlib.pyplot import show

# define meta parameters
ridge_alphas = [0.0001, 0.001, 0.01]
n_folds = 10
cutoff = 0.0

# load data
path = "/home/rgast/PycharmProjects/BrainNetworks/RC/results"
lorenz_map = pd.read_pickle(f"{path}/lorenz_map.pkl")
lorenz_data = pd.read_pickle(f"{path}/lorenz_ts.pkl")
stula_map = pd.read_pickle(f"{path}/stuart_landau_map.pkl")
stula_data = pd.read_pickle(f"{path}/stuart_landau_ts.pkl")

# train ridge regressions for each eta and alpha
etas, alphas, scores = [], [], []
for i in range(len(lorenz_map.index)):

    # extract data from dfs
    lorenz_key = lorenz_map.index[i]
    eta = lorenz_map.at[lorenz_key, 'eta']
    alpha = lorenz_map.at[lorenz_key, 'alpha']
    stula_key = stula_map.index[(stula_map.loc[:, 'alpha'] == alpha) * (stula_map.loc[:, 'eta'] == eta)]
    lorenz_ts = lorenz_data.loc[cutoff:, ("r", lorenz_key)]
    stula_ts = stula_data.loc[cutoff:, ("r", stula_key)]

    # generate training data
    X = np.concatenate((lorenz_ts.values, stula_ts.values), axis=0)
    y = np.zeros((X.shape[0],))
    y[:lorenz_data.shape[0]] = 1.0
    y[lorenz_data.shape[0]:] = -1.0

    # perform cross-validated ridge regression
    skf = StratifiedKFold(n_splits=n_folds)
    m = RidgeCV(alphas=ridge_alphas, cv=n_folds)
    scores_tmp = []
    for train_idx, test_idx in skf.split(X, y):

        # train ridge regression
        m.fit(X[train_idx], y[train_idx])

        # get score of trained model
        scores_tmp.append(m.score(X[test_idx], y[test_idx]))

    mean_score = np.mean(scores_tmp)

    # store information
    etas.append(eta)
    alphas.append(alpha)
    scores.append(mean_score)

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
