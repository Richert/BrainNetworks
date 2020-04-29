from pandas import DataFrame
import numpy as np
import seaborn as sns
from scipy.spatial.distance import pdist
from scipy.stats import zscore
import os
import matplotlib.pyplot as plt
import h5py

# parameters
directories = ["/home/rgast/JuliaProjects/JuRates/BasalGanglia/results/stn_gpe_ev_opt_results"]
fid = "stn_gpe_ev_opt"
dv = 'p'
ivs = ['eta_e', 'eta_p', 'eta_a', 'k_ee', 'k_pe', 'k_ae', 'k_ep', 'k_pp', 'k_ap', 'k_pa', 'k_aa', 'k_ps', 'k_as',
       'delta_e', 'delta_p', 'delta_a']

# load data into frame
df = DataFrame(data=np.zeros((1, len(ivs))), columns=ivs)
for d in directories:
    for fn in os.listdir(d):
        if fn.startswith(fid) and fn.endswith(".h5"):
            f = h5py.File(f"{d}/{fn}", 'r')
            index = int(fn.split('_')[-2])
            df_tmp = DataFrame(data=np.asarray([[f[dv][key][()] for key in ivs]]), columns=ivs, index=[index])
            df = df.append(df_tmp)

df = df.iloc[1:, :]

# calculate distance matrix
# visualize and cluster distance matrix
sns.clustermap(data=df, method='average', metric='correlation', z_score=1, standard_scale=None)
plt.show()
