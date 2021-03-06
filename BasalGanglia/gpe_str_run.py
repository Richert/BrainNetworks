from pyrates.utility import plot_timeseries, create_cmap
from pyrates.ir import CircuitIR
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib as mpl
plt.style.reload_library()
plt.style.use('ggplot')
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.labelcolor'] = 'black'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['ytick.color'] = 'black'
mpl.rcParams['legend.fontsize'] = 12

# parameters
dt = 1e-2
T = 500.0
dts = 1e-1
freq = 14.0
amp = 1.0
sim_steps = int(np.round(T/dt, decimals=0))
# ctx = np.zeros((sim_steps, 1))
# ctx[50000, 0] = 8000.0
# ctx = gaussian_filter1d(ctx, 100, axis=0)
time = np.linspace(0., T, sim_steps)
ctx = np.sin(2.0*np.pi*freq*time*1e-3)*amp

# plt.plot(ctx)
# plt.show()

eic = CircuitIR.from_yaml("config/stn_gpe/gpe_str").compile(backend='numpy', solver='scipy', step_size=dt)
results, t = eic.run(simulation_time=T, sampling_step_size=dts, profile=True,
                     outputs={'GPe_arky': 'gpe/gpe_arky_op/R_a',
                              'MSN': 'msn/str_msn_op/R_s',
                              'FSI': 'fsi/str_fsi_op/R_f'},
                     #inputs={'str/str_op/ctx': ctx}
                     )

results = results * 1e3
fig, ax = plt.subplots(dpi=200, figsize=(10, 3.5))
ax = plot_timeseries(results, cmap=create_cmap('cubehelix', as_cmap=False, n_colors=3), ax=ax)
plt.legend(['GPe_a', 'MSN', 'FSI'])
ax.set_title('Healthy Firing Rates')
ax.set_ylabel('firing rate (spikes/s)')
ax.set_xlabel('time (s)')
#ax.set_ylim(0.0, 100.0)
#ax.set_xlim(20.0, 240.0)
plt.tight_layout()
plt.show()
