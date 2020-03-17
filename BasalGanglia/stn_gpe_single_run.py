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
# ctx = np.zeros((int(T/dt), 1))
# ctx[150000, 0] = 1.0
# str = np.zeros((int(T/dt), 1))
# str[152200, 0] = 1.0
# ctx = gaussian_filter1d(ctx, 100, axis=0)
# str = gaussian_filter1d(str, 100, axis=0)

# plt.plot(ctx)
# plt.plot(str)
# plt.show()

eic = CircuitIR.from_yaml("config/stn_gpe/stn_gpe_basic").compile(backend='numpy', solver='scipy', step_size=dt)
results, t = eic.run(simulation_time=T, sampling_step_size=dts, profile=True,
                     outputs={'Re': 'stn/stn_op/R_e', 'Ri': 'gpe/gpe_proto_op/R_i'},
                     )
# eic = CircuitIR.from_yaml("config/stn_gpe/net_stn_gpe").compile(backend='numpy', solver='scipy', step_size=dt)
# results, t = eic.run(simulation_time=T, sampling_step_size=dts, profile=True,
#                      outputs={
#                          'Re': 'stn_gpe/qif_full/R_e',
#                          'Ri': 'stn_gpe/qif_full/R_i',
#                      },
#                      )
#eic2.generate_auto_def('config')

# eic2.set_node_var('stn_gpe/qif_driver/delta_e', 2.0)
#eic2.generate_auto_def(None)
results = results * 1e3
fig, ax = plt.subplots(dpi=200, figsize=(10, 3.5))
ax = plot_timeseries(results, cmap=create_cmap('pyrates_purple', as_cmap=False, n_colors=2), ax=ax)
plt.legend(['STN', 'GPe'])
ax.set_title('Healthy Firing Rates')
ax.set_ylabel('firing rate (spikes/s)')
ax.set_xlabel('time (s)')
#ax.set_ylim(0.0, 100.0)
#ax.set_xlim(20.0, 240.0)
plt.tight_layout()
plt.show()
