from pandas import DataFrame, read_hdf
import numpy as np
from sklearn.manifold import LocallyLinearEmbedding
import os
import matplotlib.pyplot as plt

# parameters
directories = ["/home/rgast/ownCloud/data"]
fid = "pop_summary"
dv = 'fitness'
ivs = ['eta_e', 'eta_i', 'J_ee', 'J_ei', 'J_ie', 'J_ii']

# load data into frame
df = DataFrame(data=np.zeros((1, 7)), columns=ivs + [dv])
for d in directories:
    for fn in os.listdir(d):
        if fn.startswith(fid) and fn.endswith(".h5"):
            df_tmp = read_hdf(f"{d}/{fn}")
            df = df.append(df_tmp.loc[:, ivs + [dv]])
df = df.iloc[1:, :]

# perform dimensionality reduction on data
dim_red = LocallyLinearEmbedding()
y = df.pop(dv)
X = np.asarray([df.pop(iv) for iv in ivs]).T
X_ld = dim_red.fit_transform(X, y).T

# visualize data
fig, ax = plt.subplots()
plt.scatter(X_ld[0], X_ld[1], c=y, cmap=plt.cm.rainbow)
plt.show()
