import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from pyrates.utility import functional_connectivity, plot_connectivity
from sklearn.decomposition import PCA

# load parametrizations
df = pd.read_excel('/home/rgast/ownCloud/data/stn_gpe_initial_parameters.xlsx').transpose()
df = pd.DataFrame(np.asarray(df.values[1:, :], dtype=np.float), columns=df.iloc[0 ,:])

# raw data plot
df.plot()

# cross-correlation plot
_, ax = plt.subplots()
fc = functional_connectivity(df, metric='corr')
plot_connectivity(fc, ax=ax, xticklabels=df.columns, yticklabels=df.columns)

# dimensionality reduction plot
pca = PCA(n_components=7)
reduced_data = pca.fit_transform(df.values)
_, ax2 = plt.subplots(ncols=2)
plot_connectivity(pca.components_, ax=ax2[0], xticklabels=df.columns,
                  yticklabels=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7'])
ax2[1].scatter(reduced_data[:, 0], reduced_data[:, 1])
ax2[1].set_xlabel('PC1'), ax2[1].set_ylabel('PC2')
print('Component eigenvalues: ', pca.singular_values_)
plt.tight_layout()

# choose interesting initial states
idx1 = np.argmin(np.mean((reduced_data[:, 0:2] - np.asarray([[-14.5, -1.5]]))**2, axis=1))
idx2 = np.argmin(np.mean((reduced_data[:, 0:2] - np.asarray([[12.8, 2.1]]))**2, axis=1))
idx3 = np.argmin(np.mean((reduced_data[:, 0:2] - np.asarray([[11.4, 3.6]]))**2, axis=1))
idx4 = np.argmin(np.mean((reduced_data[:, 0:2] - np.asarray([[23.4, -8.8]]))**2, axis=1))
idx5 = np.argmin(np.mean((reduced_data[:, 0:2] - np.asarray([[43.0, -0.9]]))**2, axis=1))
idx6 = np.argmin(np.mean((reduced_data[:, 0:2] - np.asarray([[20.6, 8.8]]))**2, axis=1))
idx7 = np.argmin(np.mean((reduced_data[:, 0:2] - np.asarray([[14.2, -5.5]]))**2, axis=1))

params1 = df.iloc[idx1, :]
params2 = df.iloc[idx2, :]
params3 = df.iloc[idx3, :]
params4 = df.iloc[idx4, :]
params5 = df.iloc[idx5, :]
params6 = df.iloc[idx6, :]
params7 = df.iloc[idx7, :]

print('params1: '), print(params1)
print('params2: '), print(params2)
print('params3: '), print(params3)
print('params4: '), print(params4)
print('params5: '), print(params5)
print('params6: '), print(params6)
print('params7: '), print(params7)

plt.show()
