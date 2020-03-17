from pandas import DataFrame, read_hdf
import numpy as np
from sklearn.manifold import TSNE, Isomap
from sklearn.linear_model import Ridge, LassoLars
import os
import matplotlib.pyplot as plt

# parameters
directories = ["/data/tu_rgast_cloud/owncloud-gwdg/data/stn_gpe_healthy_opt3"]
fid = "pop_summary"
dv = 'fitness'
ivs = ['eta_e', 'eta_i', 'eta_tha', 'k_str', 'k_ee', 'k_ei', 'k_ie', 'k_ii', 'delta_e', 'delta_i']
n_comps = 3

# load data into frame
df = DataFrame(data=np.zeros((1, 11)), columns=ivs + [dv])
for d in directories:
    for fn in os.listdir(d):
        if fn.startswith(fid) and fn.endswith(".h5"):
            df_tmp = read_hdf(f"{d}/{fn}")
            df = df.append(df_tmp.loc[:, ivs + [dv]])
df = df.iloc[1:, :]

# create feature matrix and target vector
y = df.pop(dv)
X = np.asarray([df.pop(iv) for iv in ivs]).T

# perform dimensionality reduction on data
dim_red = Isomap(n_components=n_comps, n_neighbors=30)
X_ld = dim_red.fit_transform(X, y)

# fit linear classifier to dim-reduced data
lm = LassoLars(alpha=1.0)
lm = lm.fit(X_ld, y)

# plot regression coefficient
fig2, ax2 = plt.subplots()
plt.bar(np.arange(0, n_comps), lm.coef_)

# visualize dim-reduced data along the two dimension with the greatest coefficients
indices = np.argsort(lm.coef_)
y -= np.min(y)
y /= np.max(y)
cmap = plt.cm.inferno
colors = cmap(y)
colors[:, 3] = y
fig, axes = plt.subplots(nrows=3)
plt.sca(axes[0])
plt.scatter(X_ld[:, indices[-1]], X_ld[:, indices[-2]], c=colors)
plt.sca(axes[1])
plt.scatter(X_ld[:, indices[-1]], X_ld[:, indices[-3]], c=colors)
plt.sca(axes[2])
plt.scatter(X_ld[:, indices[-2]], X_ld[:, indices[-3]], c=colors)
print(lm.score(X_ld, y))
plt.show()
