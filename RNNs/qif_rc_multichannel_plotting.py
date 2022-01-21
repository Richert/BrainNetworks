import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def get_peaks(x):
    peak_data = []
    for idx in range(x.shape[0]):
        s = np.abs(1 - x[idx, :])
        peaks, pinfo = find_peaks(s, width=2, distance=20, prominence=0.0005)
        if len(peaks) == 0:
            p = 0.0
        elif len(peaks) == 1:
            p = pinfo['prominences'][0]
        else:
            p1, p2 = np.sort(pinfo['prominences'])[[-1, -2]]
            p = np.abs(p1 - p2)
        peak_data.append(p)
    return np.asarray(peak_data)


data = pickle.load(open("data/qif_rc_multichannel_results.pkl", 'rb'))

# Plotting
##########

# extract data
start = 125.0
T = data["T"] - start
dt = 1e-1  # data["dts"]
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
iv_indices = np.arange(3, 8, step=1)
fig, axes = plt.subplots(nrows=len(iv_indices), ncols=2, figsize=(12, 10))
for k in range(len(iv_indices)):
    idx = iv_indices[k]

    ax1 = axes[k, 0]
    ax1.plot(times, r_qif_all[idx, :], 'orange')
    ax1.plot(times, r_mf_all[idx, :], 'blue')
    ax1.set_xlabel('time')
    ax1.set_ylabel('firing rate')
    ax1.set_title(fr'${iv_name} = {ivs[idx]}$')
    plt.legend(['QIF', 'MF'])

    ax2 = axes[k, 1]
    ax2.plot(times, Z_qif_all[idx, :], 'orange')
    ax2.plot(times, Z_mf_all[idx, :], 'blue')
    ax2.set_xlabel('time')
    ax2.set_ylabel('Z')
    ax2.set_title(fr'${iv_name} = {ivs[idx]}$')
    ax2.set_ylim([0.0, 1.0])
    plt.legend(['QIF', 'MF'])

plt.tight_layout()

# comparison between RC performance and mean-field Z-related measures
cutoff = 1000
signal = Z_mf_all[:, cutoff:-cutoff]
z_mean = np.mean(signal, axis=1)
z_max = np.max(signal, axis=1)
z_min = np.min(signal, axis=1)
z_peaks = get_peaks(Z_mf_all[:, int(start / dt):-cutoff])
z_range = z_max - z_min

zs = [z_mean, z_range, z_peaks]
labels = ['mean(Z)', 'max(Z) - min(Z)', 'PI(Z)']
colors = ['r', 'b', 'g']

fig2, axes2 = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

for i, (z, title, c) in enumerate(zip(zs, labels, colors)):
    # plot WTA score as a function of the independent variable
    ax1 = axes2[0, i]
    ax1.plot(ivs, wta, 'o:k')
    ax1.set_xlabel(f'{iv_name}')
    ax1.set_ylabel('WTA score')
    ax1.set_title(title)

    # plot Z as a function of the indipendent variable
    ax2 = ax1.twinx()
    ax2.plot(ivs, z, f'o:{c}')
    ax2.set_ylabel('Z', color=c)

    # plot WTA score as a function of Z
    ax3 = axes2[1, i]
    ax3.scatter(z, wta, color=c)
    ax3.set_xlabel('Z', color=c)
    ax3.set_ylabel('WTA score')

plt.suptitle('RNN performance vs. coherence')
plt.tight_layout()
plt.show()
