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


data = pickle.load(open("data/qif_micro_macro_results.pkl", 'rb'))

# Plotting
##########

# extract data
start = 125.0
T = data["T"] - start
dt = 1e-1  # data["dts"]
times = np.linspace(0, T, data['r_qif'].shape[1])
ivs = data["iv"]
iv_name = data["iv_name"]
Z_qif_all = data["Z_qif"]
Z_mf_all = data["Z_mf"]
r_qif_all = data['r_qif']
r_mf_all = data['r_mf']

# comparison between qif and mean-field dynamics
iv_indices = np.arange(0, 2, step=1)
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
plt.show()

# comparison of mean-field statistics
qif_signal = Z_qif_all[:]
fre_signal = Z_mf_all[:]
qif_mean = np.mean(qif_signal, axis=1)
fre_mean = np.mean(fre_signal, axis=1)
qif_max = np.max(qif_signal, axis=1)
qif_min = np.min(qif_signal, axis=1)
fre_max = np.max(fre_signal, axis=1)
fre_min = np.min(fre_signal, axis=1)

zs = [(qif_mean, fre_mean), (qif_max-qif_min, fre_max-fre_min)]
labels = ['mean(Z)', 'max(Z) - min(Z)']
colors = ['r', 'b']

fig2, axes2 = plt.subplots(nrows=len(labels), ncols=3, figsize=(12, 8))
for i, ((qif, fre), ylabel, c) in enumerate(zip(zs, labels, colors)):

    # plot QIF statistic as function of independent variable
    ax1 = axes2[i, 0]
    ax1.plot(ivs, qif, f'o:{c}')
    ax1.set_xlabel(f'{iv_name}')
    ax1.set_ylabel(ylabel)
    ax1.set_title('QIF')

    # plot FRE statistic as function of independent variable
    ax2 = axes2[i, 1]
    ax2.plot(ivs, fre, f'o:{c}')
    ax2.set_xlabel(f'{iv_name}')
    ax2.set_ylabel(ylabel)
    ax2.set_title('FRE')

    # plot QIF vs. FRE statistic
    ax3 = axes2[i, 2]
    ax3.scatter(fre, qif, color=c)
    ax3.set_xlabel('FRE')
    ax3.set_ylabel('QIF')

plt.suptitle('QIF vs. FRE steady-state activity')
plt.tight_layout()
plt.show()
