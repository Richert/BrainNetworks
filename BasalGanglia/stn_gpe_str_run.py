from pyrates.utility.visualization import plot_timeseries, create_cmap
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
T = 1000.0
dts = 1e-1
freq = 14.0
amp = 1.2
sim_steps = int(np.round(T/dt, decimals=0))
# ctx = np.zeros((sim_steps, 1))
# ctx[50000, 0] = 600.0
# ctx = gaussian_filter1d(ctx, 100, axis=0)
time = np.linspace(0., T, sim_steps)
ctx = np.sin(2.0*np.pi*freq*time*1e-3)*amp

#plt.plot(ctx)
#plt.show()

eic = CircuitIR.from_yaml("config/stn_gpe_str/stn_gpe_str").compile(backend='numpy', solver='scipy', step_size=dt)
results, t = eic.run(simulation_time=T, sampling_step_size=dts, profile=True,
                     outputs={'stn': 'stn/stn_op/R',
                              'gpe-p': 'gpe_p/stn_op/R',
                              'gpe-a': 'gpe_a/stn_op/R',
                              'msn-d1': 'msn_d1/stn_op/R',
                              'msn-d2': 'msn_d2/stn_op/R',
                              'fsi-d1': 'fsi_d1/fsi_op/R',
                              'fsi-d2': 'fsi_d2/fsi_op/R'
                              },
                     # inputs={'stn/stn_op/ctx': amp + ctx,
                     #         'msn/str_msn_op/ctx': amp + ctx,
                     #         'fsi/str_fsi_op/ctx': amp + ctx}
                     )

results = results * 1e3
fig, axes = plt.subplots(nrows=2, dpi=200, figsize=(10, 5))
ax = plot_timeseries(results, cmap=create_cmap('cubehelix', as_cmap=False, n_colors=7), ax=axes[0])
#plt.legend(['STN', 'GPe_p', 'GPe_a', 'STR'])
ax.set_title('Healthy Firing Rates')
ax.set_ylabel('firing rate (spikes/s)')
ax.set_xlabel('time (ms)')
#ax.set_ylim(0.0, 100.0)
#ax.set_xlim(20.0, 240.0)
# av_signal = results.loc[:, ('STN', 'stn')] - results.loc[:, ('GPe_proto', 'gpe_p')] - results.loc[:, ('MSN', 'msn')]
# ax = plot_timeseries(av_signal, cmap=create_cmap('cubehelix', as_cmap=False, n_colors=1), ax=axes[1])
# ax.set_title('GPi input (STN - GPe_p - MSN)')
# ax.set_ylabel('firing rate (spikes/s)')
# ax.set_xlabel('time (ms)')
plt.tight_layout()
plt.show()
