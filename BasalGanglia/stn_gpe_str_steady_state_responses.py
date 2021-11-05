import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from pyrates.frontend import CircuitTemplate, clear_frontend_caches
from copy import deepcopy

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

# parameter definitions
#######################

# simulation parameters
dt = 1e-3
dts = 1e-1
T = 2000.0

# coupling strengths
k = 0.1
k_pp = 1.0*k
k_ap = 1.0*k
k_sp = 1.0*k
k_fp = 1.0*k

k_d1a = 1.0*k
k_d2a = 1.0*k

k_ps = 1.0*k
k_as = 1.0*k

k_pd2 = 1.0*k
k_ad1 = 1.0*k
k_d1d1 = 1.0*k
k_d1d2 = 1.0*k
k_d2d2 = 1.0*k

k_d1f = 1.0*k
k_d2f = 1.0*k
k_ff = 1.0*k

# excitabilities
eta_p = 12.0
eta_a = 2.0
eta_s = 0.7
eta_d1 = -1.0
eta_d2 = 1.0
eta_f = 0.5

# different input conditions and their parameter changes
conditions = [{},  # healthy control -> GPe-p: 30 Hz, GPe-a: 3 Hz, STN: 6 Hz
              {'msn_d2/msn_d2_op/eta': 30.0},  # STR excitation -> GPe-p: 3 Hz, GPe-a: 25 Hz, STN: 18 Hz
              {'stn/stn_op/eta': -20.0},  # STN inhibition -> GPe-p: 10 Hz, GPe_a: 12 Hz, STN: 1 Hz
              {'gpe_p/gpe_p_op/eta': 20.0},  # GPe-p excitation -> GPe-p: 100 Hz, GPe-a: 25 Hz
              ]

# parameter mapping
node_updates = {'gpe_p/gpe_p_op/eta': eta_p,
                'gpe_a/gpe_a_op/eta': eta_a,
                'stn/stn_op/eta': eta_s,
                'msn_d1/msn_d1_op/eta': eta_d1,
                'msn_d2/msn_d2_op/eta': eta_d2,
                'fsi/fsi_op/eta': eta_f,
                }
edge_updates = []

# simulations
#############

outputs = {
            'stn': 'stn/stn_op/R',
            'gpe-p': 'gpe_p/gpe_p_op/R',
            'gpe-a': 'gpe_a/gpe_a_op/R',
            'msn-d1': 'msn_d1/msn_d1_op/R',
            'msn-d2': 'msn_d2/msn_d2_op/R',
            'fsi': 'fsi/fsi_op/R',
        }

for c_dict in deepcopy(conditions):

    model = CircuitTemplate.from_yaml("config/stn_gpe_str/stn_gpe_str")
    model = model.update_var(node_vars=node_updates, edge_vars=edge_updates)
    results = model.run(simulation_time=T, step_size=dt, sampling_step_size=dts, outputs=outputs.copy(), solver='scipy',
                        verbose=True, method='RK23', atol=1e-5, rtol=1e-4, clear=True)
    clear_frontend_caches()

    fig, ax = plt.subplots(figsize=(6, 2.0), dpi=dpi)
    results = results * 1e3
    for key in outputs:
        ax.plot(results.loc[:, key])
    plt.legend(list(outputs.keys()))
    ax.set_ylabel('Firing rate')
    ax.set_xlabel('time (ms)')
    # ax.set_xlim([4000.0, 5000.0])
    # ax.set_ylim([0.0, 50.0])
    ax.tick_params(axis='both', which='major', labelsize=9)
    plt.tight_layout()
    plt.show()
