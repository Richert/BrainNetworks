from pyrates.utility.pyauto import PyAuto
import matplotlib.pyplot as plt
import os
import numpy as np

fig, ax = plt.subplots(figsize=(6, 4))
a = PyAuto(f"{os.getcwd()}/config")
s, c = a.run(e='gpe', c='gpe', ICP=[3], name='eta', RL0=-20.0, RL1=50.0, bidirectional=True, NMX=6000, DSMAX=0.5,
             STOP={})
a.plot_continuation('PAR(3)', 'U(1)', cont='eta', ax=ax)
ax.set_xlabel(r'$\eta$')
ax.set_ylabel('r in 1/s')
fig.canvas.draw()
labels = ax.get_yticklabels()
ax.set_yticklabels([np.round(label._y, decimals=3)*1e3 for label in labels])
plt.savefig(f'../results/gpe_cont.svg')
plt.show()
