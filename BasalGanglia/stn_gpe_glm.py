from pandas import DataFrame, read_hdf
import numpy as np
from sklearn.manifold import TSNE, Isomap
from sklearn.linear_model import Ridge, LassoLars
import os
import matplotlib.pyplot as plt
import h5py

# parameters
directories = ["/home/rgast/JuliaProjects/JuRates/BasalGanglia/results/stn_gpe_ev_opt_results_final"]
fid = "stn_gpe_ev_opt"
dv = 'p'
ivs = ['eta_e', 'eta_p', 'eta_a', 'k_ee', 'k_pe', 'k_ae', 'k_ep', 'k_pp', 'k_ap', 'k_pa', 'k_aa', 'k_ps', 'k_as',
       'delta_e', 'delta_p', 'delta_a']

# load data into frame
df = DataFrame(data=np.zeros((1, len(ivs))), columns=ivs)
df_dv = DataFrame(data=np.zeros((1, 1)), columns=["fitness"])
for d in directories:
    for fn in os.listdir(d):
        if fn.startswith(fid):
            f = h5py.File(f"{d}/{fn}", 'r')
            index = int(fn.split('_')[-2])
            if fn.endswith("params.h5"):
                df_tmp = DataFrame(data=np.asarray([[f[dv][key][()] for key in ivs]]), columns=ivs, index=[index])
                df = df.append(df_tmp)
            elif fn.endswith("fitness.h5"):
                df_tmp = DataFrame(data=np.asarray([1/f["f"][()]]), columns=["fitness"], index=[index])
                df_dv = df_dv.append(df_tmp)
df = df.iloc[1:, :]
df_dv = df_dv.iloc[1:, :]
#df['fitness'] = df_dv.loc[:, "fitness"]

# create feature matrix and target vector
y = np.squeeze(df_dv.values)
X = np.asarray([df.pop(iv) for iv in ivs]).T

# perform dimensionality reduction on data
n_comps = 5
dim_red = Isomap(n_components=n_comps, n_neighbors=10, p=2)
X_ld = dim_red.fit_transform(X, y)

# fit linear classifier to dim-reduced data
X_final = X
lm = LassoLars(alpha=0.0)
lm = lm.fit(X_final, y)

# plot regression coefficient
fig2, ax2 = plt.subplots()
plt.bar(ivs, lm.coef_)
plt.tight_layout()

# visualize dim-reduced data along the two dimension with the greatest coefficients
indices = np.argsort(np.abs(lm.coef_))
y_tmp = y - np.min(y)
y_tmp /= np.max(y_tmp)
cmap = plt.cm.inferno
colors = cmap(y_tmp)
#colors[:, 3] = y_tmp
fig, axes = plt.subplots(nrows=3)
plt.sca(axes[0])
plt.scatter(X_final[:, indices[-1]], X_final[:, indices[-2]], c=colors)
axes[0].set_xlabel(ivs[indices[-1]])
axes[0].set_ylabel(ivs[indices[-2]])
plt.sca(axes[1])
plt.scatter(X_final[:, indices[-1]], X_final[:, indices[-3]], c=colors)
axes[1].set_xlabel(ivs[indices[-1]])
axes[1].set_ylabel(ivs[indices[-3]])
plt.sca(axes[2])
plt.scatter(X_final[:, indices[-2]], X_final[:, indices[-3]], c=colors)
axes[2].set_xlabel(ivs[indices[-2]])
axes[2].set_ylabel(ivs[indices[-3]])
plt.tight_layout()
print(lm.score(X_final, y))
plt.show()
