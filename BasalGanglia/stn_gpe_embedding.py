from pandas import DataFrame, read_hdf
import numpy as np
from sklearn.manifold import MDS, Isomap
import os
import matplotlib.pyplot as plt

# parameters
directories = ["/data/tu_rgast_cloud/owncloud-gwdg/data/stn_gpe_healthy_opt3"]
fid = "pop_summary"
dv = 'fitness'
ivs = ['eta_e', 'eta_i', 'eta_tha', 'k_str', 'k_ee', 'k_ei', 'k_ie', 'k_ii', 'delta_e', 'delta_i']
n_comps = 2

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

# visualize dim-reduced data along the two dimension with the greatest coefficients
fig, ax = plt.subplots()
plt.scatter(X_ld[:, 0], X_ld[:, 1], c=y, cmap=plt.cm.cividis)

plt.show()
