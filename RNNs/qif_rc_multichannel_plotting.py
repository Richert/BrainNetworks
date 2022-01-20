import numpy as np
import pickle
import matplotlib.pyplot as plt

data = pickle.load(open("../RC/Results/qif_rc_multichannel_results.pkl", 'rb'))

# Plotting
##########

# extract data
T = data["T"]
times = np.linspace(0, T, data['r_qif'].shape[1])
ivs = data["iv"]
iv_name = data["iv_name"]
wta = data["wta_score"]
score = data["score"]
Z_qif_all = data["Z_qif"]
Z_mf_all = data["Z_mf"]
r_qif_all = data['r_qif']
r_mf_all = data['r_mf']

# comparison between qif and mean-field dynamics
fig, axes = plt.subplots(nrows=len(ivs), ncols=2, figsize=(12, 10))
for k in range(len(ivs)):

    ax1 = axes[k, 0]
    ax1.plot(times, r_qif_all[k, :], 'orange')
    ax1.plot(times, r_mf_all[k, :], 'blue')
    ax1.set_xlabel('time')
    ax1.set_ylabel('firing rate')
    ax1.set_title(fr'${iv_name} = {ivs[k]}$')
    plt.legend(['QIF', 'MF'])

    ax2 = axes[k, 1]
    ax2.plot(times, Z_qif_all[k, :], 'orange')
    ax2.plot(times, Z_mf_all[k, :], 'blue')
    ax2.set_xlabel('time')
    ax2.set_ylabel('Z')
    ax2.set_title(fr'${iv_name} = {ivs[k]}$')
    ax2.set_ylim([0.0, 1.0])
    plt.legend(['QIF', 'MF'])

plt.tight_layout()

# comparison between RC performance and KMO
z_mean = np.mean(Z_qif_all, axis=1)
z_max = np.max(Z_qif_all, axis=1)
z_min = np.min(Z_qif_all, axis=1)
z_std = np.std(Z_qif_all, axis=1)

zs = [z_mean, z_max - z_min, z_std]
labels = ['mean(Z)', 'max(Z) - min(Z)', 'std(Z)']
colors = ['red', 'orange', 'purple']

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

for i, (z, title, c) in enumerate(zip(zs, labels, colors)):

    # plot WTA score as a function of the independent variable
    ax1 = axes[0, i]
    ax1.plot(ivs, wta, 'blue')
    ax1.set_xlabel(f'{iv_name}')
    ax1.set_ylabel('WTA score', color='blue')
    ax1.set_title(title)

    # plot Z as a function of the indipendent variable
    ax2 = ax1.twinx()
    ax2.plot(ivs, z, c)
    ax2.set_ylabel('Z', color=c)

    # plot WTA score as a function of Z
    ax3 = axes[1, i]
    ax3.scatter(z, wta, color=c)
    ax3.set_xlabel('Z', color=c)
    ax3.set_ylabel('WTA score')

plt.suptitle('RNN performance vs. coherence')
plt.tight_layout()
plt.show()
