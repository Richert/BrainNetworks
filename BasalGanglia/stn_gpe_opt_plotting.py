from pandas import DataFrame, read_hdf
import numpy as np
import os
import matplotlib.pyplot as plt

# parameters
directories = ["/data/u_rgast_software/PycharmProjects/BrainNetworks/BasalGanglia/stn_gpe_healthy_opt2/PopulationDrops"]
fid = "PopulationDrop"
params = ['eta_e', 'eta_p', 'eta_a', 'delta_e', 'delta_p', 'delta_a', 'tau_e', 'tau_p', 'tau_a',
          'k_ee', 'k_pe', 'k_ae', 'k_ep', 'k_pp', 'k_ap', 'k_pa', 'k_aa', 'k_ps', 'k_as']
result_vars = ['results', 'fitness', 'sigma']

# load data into frame
df = DataFrame(data=np.zeros((1, len(params) + len(result_vars))), columns=result_vars + params)
for d in directories:
    for fn in os.listdir(d):
        if fn.startswith(fid) and fn.endswith(".h5"):
            df = df.append(read_hdf(f"{d}/{fn}"))
df = df.iloc[1:, :]

# plot histograms of conditions
winner = df.nlargest(n=1, columns='fitness')
fig, ax = plt.subplots()

labels = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
model_rates = np.round(winner['results'].iloc[0][1], decimals=1)
target_rates = [60.0, 40.0, 70.0, 100.0, 30.0, 60.0, 100.0]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

rects1 = ax.bar(x - width / 2, model_rates, width, label='model')
rects2 = ax.bar(x + width / 2, target_rates, width, label='target')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('GPe firing rate')
ax.set_title('Comparison between model fit and target firing rates')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()
