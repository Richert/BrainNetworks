from pandas import DataFrame
import numpy as np
import seaborn as sns
from scipy.spatial.distance import pdist
from scipy.stats import zscore
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
df['fitness'] = df_dv.loc[:, "fitness"]

# visualize and cluster distance matrix
sns.clustermap(data=df, method='ward', metric='euclidean', z_score=1, standard_scale=None)
#sns.clustermap(data=df_dv, method='ward', metric='euclidean', z_score=1, standard_scale=None, col_cluster=False,
#               row_cluster=True)
plt.show()
