# my_cgs_worker.py
from pyrates.utility.grid_search import ClusterWorkerTemplate
import os
from pandas import DataFrame
from pyrates.utility import grid_search, welch
import numpy as np
from copy import deepcopy


class MinimalWorker(ClusterWorkerTemplate):
    def worker_postprocessing(self, **kwargs):
        self.processed_results = DataFrame(data=None, columns=self.results.columns)
        for idx, data in self.results.iteritems():
            self.processed_results.loc[:, idx] = data * 1e3
        self.processed_results.index = self.results.index * 1e-3


class ExtendedWorker(MinimalWorker):
    def worker_gs(self, *args, **kwargs):
        kwargs_tmp = deepcopy(kwargs)
        param_grid = kwargs_tmp.pop('param_grid')
        r, self.result_map, sim_time = grid_search(*args, param_grid=param_grid, **kwargs_tmp)
        r = r.droplevel(2, axis=1)
        self.results = r
        return sim_time

    def worker_postprocessing(self, **kwargs):
        kwargs_tmp = kwargs.copy()
        param_grid = kwargs_tmp.pop('param_grid')
        targets = kwargs_tmp.pop('targets')
        self.processed_results = DataFrame(data=None, columns=['fitness', 'frequency', 'power', 'r_e', 'r_i'])

        # calculate fitness
        for gene_id in param_grid.index:
            r = self.results
            r = r * 1e3
            r.index = r.index * 1e-3
            cutoff = r.index[-1]*0.1
            mean_re = np.mean(r['r_e'][f'circuit_{gene_id}'].loc[cutoff:])
            mean_ri = np.mean(r['r_i'][f'circuit_{gene_id}'].loc[cutoff:])
            outputs = [mean_re, mean_ri]
            psds, freqs = welch(r['r_i'][f'circuit_{gene_id}'], tmin=cutoff, fmin=5.0, fmax=200.0)

            dist1 = fitness(outputs, targets)
            dist2 = analyze_oscillations([0.0], [freqs], [psds])
            idx = np.argmax(psds[0])
            r = self.results[0]
            self.processed_results.loc[gene_id, 'fitness'] = dist1+dist2
            self.processed_results.loc[gene_id, 'frequency'] = freqs[idx]
            self.processed_results.loc[gene_id, 'power'] = psds[0][idx]
            self.processed_results.loc[gene_id, 'r_e'] = np.mean(r['r_e'][f'circuit_{gene_id}'].loc[cutoff:])*1e3
            self.processed_results.loc[gene_id, 'r_i'] = np.mean(r['r_i'][f'circuit_{gene_id}'].loc[cutoff:])*1e3


def fitness(y, t):
    y = np.asarray(y).flatten()
    t = np.asarray(t).flatten()
    diff = np.asarray([0.0 if np.isnan(t_tmp) else y_tmp - t_tmp for y_tmp, t_tmp in zip(y, t)])
    return np.sqrt(np.mean(diff**2))


def analyze_oscillations(freq_targets, freqs, pows):
    dist = []
    for i, (t, f, p) in enumerate(zip(freq_targets, freqs, pows)):
        if type(t) is list:
            f_tmp = f[np.argmax(p)]
            dist.append(f_tmp)
            if f_tmp < t[0]:
                freq_targets[i] = t[0]
            elif f_tmp > t[1]:
                freq_targets[i] = t[1]
            else:
                freq_targets[i] = f_tmp
        elif np.isnan(t):
            dist.append(0.0)
        elif t:
            f_tmp = f[np.argmax(p)]
            dist.append(f_tmp)
        else:
            p_tmp = np.max(p)
            dist.append(p_tmp)
    return fitness(dist, freq_targets)


if __name__ == "__main__":
    cgs_worker = ExtendedWorker()
    cgs_worker.worker_init()
    #cgs_worker.worker_init(
    #    config_file="/nobackup/spanien1/rgast/PycharmProjects/BrainNetworks/BasalGanglia/stn_gpe_optimization/Config/DefaultConfig_0.yaml",
    #    subgrid="/nobackup/spanien1/rgast/PycharmProjects/BrainNetworks/BasalGanglia/stn_gpe_optimization/Grids/Subgrids/DefaultGrid_11/spanien/spanien_Subgrid_2.h5",
    #    result_file="~/my_result.h5",
    #    build_dir=os.getcwd()
    #)
