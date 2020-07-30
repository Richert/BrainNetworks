import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from pyrates.utility.visualization import create_cmap, plot_connectivity
from scipy.interpolate.interpolate import interp1d
import scipy.io as scio

linewidth = 1.2
fontsize1 = 10
fontsize2 = 10
markersize1 = 60
markersize2 = 60
dpi = 200

plt.style.reload_library()
plt.style.use('seaborn-whitegrid')
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']
#mpl.rc('text', usetex=True)
mpl.rcParams["font.sans-serif"] = ["Roboto"]
mpl.rcParams["font.size"] = fontsize1
mpl.rcParams["font.weight"] = "bold"
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rcParams['axes.titlesize'] = fontsize2
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['axes.labelsize'] = 'large'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['ytick.color'] = 'black'
mpl.rcParams['ytick.alignment'] = 'center'
mpl.rcParams['legend.fontsize'] = fontsize1
sns.set(style="whitegrid")

# load data
path = "/home/rgast/MatlabProjects/STN_GPe/PAC_bistable.mat"
data = scio.loadmat(path)

# extract from data
alpha = data['stim_amps_unique']
omega = data['stim_freqs_unique']
MI = data['PAC_max']
PAA_osc = data['PAA_osc']
PAA_env = data['PAA_env']
PAA = PAA_osc / PAA_env

# plot MI
cmap = create_cmap("pyrates_blue", n_colors=64, as_cmap=False, reverse=False)
ax = plot_connectivity(MI, cmap=cmap)
ax.set_xticks(ax.get_xticks()[0::2])
ax.set_yticks(ax.get_yticks()[0::2])
ax.set_xticklabels(np.round(omega.squeeze(), decimals=1)[0::2], rotation='horizontal')
ax.set_yticklabels(np.round(alpha.squeeze(), decimals=1)[0::2], rotation='horizontal')
ax.set_xlabel('omega')
ax.set_ylabel('alpha')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('MI_bs.svg')
plt.show()

# plot PAA
cmap = create_cmap("pyrates_blue", n_colors=64, as_cmap=False, reverse=False)
ax = plot_connectivity(PAA, cmap=cmap, vmin=0.0, vmax=1.0)
ax.set_xticks(ax.get_xticks()[0::2])
ax.set_yticks(ax.get_yticks()[0::2])
ax.set_xticklabels(np.round(omega.squeeze(), decimals=1)[0::2], rotation='horizontal')
ax.set_yticklabels(np.round(alpha.squeeze(), decimals=1)[0::2], rotation='horizontal')
ax.set_xlabel('omega')
ax.set_ylabel('alpha')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('PAA_bs.svg')
plt.show()
