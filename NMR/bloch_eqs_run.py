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
T = 1000.0
dts = 1e-1
freq = 14.0
amp = 4.0
sim_steps = int(np.round(T/dt, decimals=0))
inp = np.zeros((sim_steps, 1))
inp[20000, 0] = 100.0
inp = gaussian_filter1d(inp, 100, axis=0)
#time = np.linspace(0., T, sim_steps)
#inp = np.sin(2.0*np.pi*freq*time*1e-3)*amp

# plt.plot(ctx)
plt.plot(inp)
plt.show()

bloch = CircuitIR.from_yaml("nmr/net"
                            ).compile(backend='numpy', solver='scipy', step_size=dt)
results, t = bloch.run(simulation_time=T, sampling_step_size=dts, profile=True,
                       outputs={'x': 'n1/bloch/mx', 'y': 'n1/bloch/my'},
                       inputs={'n1/bloch/x_in': inp, 'n1/bloch/y_in': inp}
                       )

results = results * 1e3
fig, ax = plt.subplots(dpi=200, figsize=(10, 3.5))
ax = plot_timeseries(results, cmap=create_cmap('pyrates_purple', as_cmap=False, n_colors=2), ax=ax)
#plt.legend(['STN', 'GPe_p', 'GPe_a'])
ax.set_title('Nuclear Spin')
ax.set_ylabel('coordinate')
ax.set_xlabel('time')
plt.tight_layout()
plt.show()
