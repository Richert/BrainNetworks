import numpy as np
import pickle
import matplotlib.pyplot as plt

data = pickle.load(open("qif_rc_multichannel_results.pkl", 'rb'))
T = 625.0

# Plotting
##########

# extract data
times = np.linspace(0, T, data['r_qif'].shape[1])
etas = data["etas"]
wta = data["wta_score"]
score = data["score"]
Z_qif_all = data["Z_qif"]
Z_mf_all = data["Z_mf"]
r_qif_all = data['r_qif']
r_mf_all = data['r_mf']

# comparison between qif and mean-field dynamics
fig, axes = plt.subplots(nrows=len(etas), ncols=2, figsize=(12, 10))
for k in range(len(etas)):

    ax1 = axes[k, 0]
    ax1.plot(times, r_qif_all[k, :], 'orange')
    ax1.plot(times, r_mf_all[k, :], 'blue')
    ax1.set_xlabel('time')
    ax1.set_ylabel('firing rate')
    ax1.set_title(fr'$\eta = {etas[k]}$')
    plt.legend(['QIF', 'MF'])

    ax2 = axes[k, 1]
    ax2.plot(times, Z_qif_all[k, :], 'orange')
    ax2.plot(times, Z_mf_all[k, :], 'blue')
    ax2.set_xlabel('time')
    ax2.set_ylabel('Z')
    ax2.set_title(fr'$\eta = {etas[k]}$')
    plt.legend(['QIF', 'MF'])

plt.tight_layout()

# comparison between RC performance and KMO
fig, axes = plt.subplots(nrows=3, figsize=(12, 8))

z_mean = np.mean(Z_qif_all, axis=1)
z_max = np.max(Z_qif_all, axis=1)

ax1 = axes[0]
ax1.plot(etas, wta)
ax1.set_xlabel(r'$\eta$')
ax1.set_ylabel('WTA score')
ax1.set_title('RNN classification performance')

ax2 = axes[1]
ax2.plot(etas, z_max)
ax2.set_xlabel(r'$\eta$')
ax2.set_ylabel('Z')
ax2.set_title('RNN coherence')

ax3 = axes[2]
ax3.scatter(z_max, wta)
ax3.set_xlabel('Z')
ax3.set_ylabel('WTA score')
ax3.set_title('RNN performance vs. coherence')
plt.tight_layout()

plt.show()