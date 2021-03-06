from pandas import DataFrame
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
import h5py
from pyrates.utility.visualization import create_cmap

# parameters
directories = ["/home/rgast/JuliaProjects/JuRates/BasalGanglia/results/stn_gpe_beta_results"]
fid = "stn_gpe_beta"
dv = 'p'
ivs = ["tau_e", "tau_p", "tau_ampa_r", "tau_ampa_d", "tau_gabaa_r", "tau_gabaa_d", "tau_stn", "eta_e", "eta_p",
       "delta_e", "delta_p", "k_pe", "k_ep", "k_pp"]

# load data into frame
df = DataFrame(data=np.zeros((1, len(ivs))), columns=ivs)
df_dv = DataFrame(data=np.zeros((1, 1)), columns=["fitness"])
for d in directories:
    for fn in os.listdir(d):
        if fn.startswith(fid) and fn.endswith('.h5'):
            f = h5py.File(f"{d}/{fn}", 'r')
            index = int(fn.split('_')[-1][:-3])
            df_tmp = DataFrame(data=np.asarray([[f[dv][key][()] for key in ivs]]), columns=ivs, index=[index])
            df = df.append(df_tmp)
            df_tmp2 = DataFrame(data=np.asarray([1/f["f/f"][()]]), columns=["fitness"], index=[index])
            df_dv = df_dv.append(df_tmp2)
df = df.iloc[1:, :]
df_dv = df_dv.iloc[1:, :]
df.sort_index(inplace=True)
df_dv.sort_index(inplace=True)
df['fitness'] = df_dv['fitness']

# visualize and cluster distance matrix
cmap = create_cmap('pyrates_red', as_cmap=True, n_colors=20)
g = sns.clustermap(data=df, method='ward', metric='euclidean', z_score=1, standard_scale=None, cmap=cmap)
#sns.clustermap(data=df_dv, method='ward', metric='euclidean', z_score=1, standard_scale=None, col_cluster=False,
#               row_cluster=True)
alphas = np.asarray([np.squeeze(df_dv.values) for _ in range(len(ivs))]).T
alphas /= np.max(alphas)
#g.ax_heatmap.set_alpha(alphas)
plt.show()
