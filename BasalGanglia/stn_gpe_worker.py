# my_cgs_worker.py
from pyrates.utility.grid_search import ClusterWorkerTemplate
import os
from pandas import DataFrame
from pyrates.utility import grid_search, welch
import numpy as np


class MinimalWorker(ClusterWorkerTemplate):
    def worker_postprocessing(self, **kwargs):
        self.processed_results = DataFrame(data=None, columns=self.results.columns)
        for idx, data in self.results.iteritems():
            self.processed_results.loc[:, idx] = data * 1e3
        self.processed_results.index = self.results.index * 1e-3


class ExtendedWorker(MinimalWorker):
    def worker_gs(self, *args, **kwargs):
        conditions = kwargs.pop('conditions')
        model_vars = kwargs.pop('model_vars')
        param_grid = kwargs.pop('param_grid')
        results = []
        t = 0
        for c_dict in conditions:
            param_grid_tmp = {key: param_grid[key] for key in model_vars}.copy()
            param_grid_tmp.update(DataFrame(c_dict, index=param_grid.index))
            r, self.result_map, t_tmp = grid_search(*args, param_map=param_grid_tmp, **kwargs)
            results.append(r)
            t += t_tmp
        self.results = results
        return t

    def worker_postprocessing(self, **kwargs):

        param_grid = kwargs.pop('param_grid')
        freq_targets = kwargs.pop('freq_targets')
        targets = kwargs.pop('targets')
        self.processed_results = DataFrame(data=None, columns=['fitness', 'frequency', 'power', 'r_e', 'r_i'])

        # calculate fitness
        for gene_id in param_grid.index:
            outputs, freq, pow = [], [], []
            for i, r in enumerate(self.results):
                if r is None:
                    outputs.append([np.inf, np.inf])
                    freq.append([0.0])
                    pow.append([0.0])
                else:
                    r = r * 1e3
                    r.index = r.index * 1e-3
                    mean_re = np.mean(r['r_e'][f'circuit_{gene_id}'].loc[2.0:])
                    mean_ri = np.mean(r['r_i'][f'circuit_{gene_id}'].loc[2.0:])
                    outputs.append([mean_re, mean_ri])
                    tmin = 0.0 if i == 4 else 2.0
                    psds, freqs = welch(r['r_i'][f'circuit_{gene_id}'], tmin=tmin, fmin=5.0, fmax=100.0)
                    freq.append(freqs)
                    pow.append(psds)

            dist1 = fitness(outputs, targets)
            dist2 = analyze_oscillations(freq_targets, freq, pow)
            idx = np.argmax(pow)[0]
            self.processed_results.loc[gene_id, 'fitness'] = dist1+dist2
            self.processed_results.loc[gene_id, 'freq'] = freq[idx]
            self.processed_results.loc[gene_id, 'pow'] = pow[idx]
            self.processed_results.loc[gene_id, 'r_e'] = np.mean(self.results[0]['r_e'][f'circuit_{gene_id}'].loc[2.0:])
            self.processed_results.loc[gene_id, 'r_i'] = np.mean(self.results[0]['r_i'][f'circuit_{gene_id}'].loc[2.0:])


def fitness(y, t):
    y = np.asarray(y).flatten()
    t = np.asarray(t).flatten()
    diff = np.asarray([0.0 if np.isnan(t_tmp) else y_tmp - t_tmp for y_tmp, t_tmp in zip(y, t)])
    return np.sqrt(np.mean(diff**2))


def analyze_oscillations(freq_targets, freqs, pows):
    dist = []
    for t, f, p in zip(freq_targets, freqs, pows):
        if np.isnan(t):
            dist.append(0.0)
        elif t:
            f_tmp = f[np.argmax(p)]
            dist.append(t - f_tmp)
        else:
            p_tmp = np.max(p)
            dist.append(-p_tmp)
    return fitness(dist, freq_targets)


if __name__ == "__main__":
    cgs_worker = MinimalWorker()
    #cgs_worker.worker_init()
    cgs_worker.worker_init(
        config_file="/nobackup/spanien1/rgast/PycharmProjects/BrainNetworks/BasalGanglia/results/Config/DefaultConfig_0.yaml",
        subgrid="/nobackup/spanien1/rgast/PycharmProjects/BrainNetworks/BasalGanglia/results/Grids/Subgrids/DefaultGrid_691/animals/animals_Subgrid_0.h5",
        result_file="~/my_result.h5",
        build_dir=os.getcwd()
    )
