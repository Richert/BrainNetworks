from pandas import DataFrame, read_hdf
import numpy as np
from sklearn.manifold import TSNE, Isomap
from sklearn.linear_model import Ridge, LassoLars
import os
import matplotlib.pyplot as plt

# parameters
directories = ["/home/rgast/ownCloud/data/stn_gpe_healthy_opt3"]
fid = "pop_summary"
dv = 'fitness'
ivs = ['eta_e', 'eta_i', 'eta_tha', 'k_str', 'k_ee', 'k_ei', 'k_ie', 'k_ii', 'delta_e', 'delta_i']
n_comps = 6

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
lm = Ridge(alpha=0.1)
lm = lm.fit(X_ld, y)

# plot regression coefficient
fig2, ax2 = plt.subplots()
plt.bar(np.arange(0, n_comps), lm.coef_)

# visualize dim-reduced data along the two dimension with the greatest coefficients
indices = np.argsort(lm.coef_)
fig, ax = plt.subplots()
plt.scatter(X_ld[:, indices[-1]], X_ld[:, indices[-2]], c=y, cmap=plt.cm.inferno)

print(lm.score(X_ld, y))
plt.show()
