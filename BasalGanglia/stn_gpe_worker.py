# my_cgs_worker.py
from pyrates.utility.grid_search import ClusterWorkerTemplate
import os
from pandas import DataFrame, concat, MultiIndex
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
        kwargs_tmp = kwargs.copy()
        conditions = kwargs_tmp.pop('conditions')
        model_vars = kwargs_tmp.pop('model_vars')
        param_grid = kwargs_tmp.pop('param_grid')
        results, gene_ids = [], param_grid.index
        for c_dict in conditions:
            for key in model_vars:
                if key in c_dict and type(c_dict[key]) is float:
                    c_dict[key] = np.zeros((param_grid.shape[0],)) + c_dict[key]
                else:
                    c_dict[key] = param_grid[key]
            param_grid_tmp = DataFrame.from_dict(c_dict)
            f = terminate_at_threshold
            f.terminal = True
            r, self.result_map, sim_time = grid_search(*args, param_grid=param_grid_tmp, events=f, **kwargs_tmp)
            r.columns.droplevel(2)
            if any(r.values[-1, :] > 1.0):
                invalid_genes = []
                for id in gene_ids:
                    if r.loc[-1, ('r_e', id)] > 1.0 or r.loc[-1, ('r_i', id)] > 1.0:
                        invalid_genes.append(id)
                        param_grid.pop(id)
                kwargs['param_grid'] = param_grid
                sim_time = self.worker_gs(*args, **kwargs)
                for r in self.results:
                    for id in invalid_genes:
                        r[('r_e', id)] = np.zeros((r.shape[0],)) + np.inf
                        r[('r_i', id)] = np.zeros((r.shape[0],)) + np.inf
                return sim_time
            else:
                results.append(r)
        self.results = results
        return sim_time

    def worker_postprocessing(self, **kwargs):
        kwargs_tmp = kwargs.copy()
        param_grid = kwargs_tmp.pop('param_grid')
        freq_targets = kwargs_tmp.pop('freq_targets')
        targets = kwargs_tmp.pop('targets')
        self.processed_results = DataFrame(data=None, columns=['fitness', 'frequency', 'power', 'r_e', 'r_i'])

        # calculate fitness
        for gene_id in param_grid.index:
            outputs, freq, pow = [], [], []
            for i, r in enumerate(self.results):
                r = r * 1e3
                r.index = r.index * 1e-3
                cutoff = r.index[-1]*0.1
                mean_re = np.mean(r['r_e'][f'circuit_{gene_id}'].loc[cutoff:])
                mean_ri = np.mean(r['r_i'][f'circuit_{gene_id}'].loc[cutoff:])
                outputs.append([mean_re, mean_ri])
                tmin = 0.0 if i == 4 else cutoff
                psds, freqs = welch(r['r_i'][f'circuit_{gene_id}'], tmin=tmin, fmin=5.0, fmax=100.0)
                freq.append(freqs)
                pow.append(psds)

            dist1 = fitness(outputs, targets)
            dist2 = analyze_oscillations(freq_targets, freq, pow)
            idx = np.argmax(pow[-1][0])
            r = self.results[0]
            cutoff = r.index[-1]*0.1
            self.processed_results.loc[gene_id, 'fitness'] = dist1+dist2
            self.processed_results.loc[gene_id, 'frequency'] = freq[-1][idx]
            self.processed_results.loc[gene_id, 'power'] = pow[-1][0][idx]
            self.processed_results.loc[gene_id, 'r_e'] = np.mean(r['r_e'][f'circuit_{gene_id}'].loc[cutoff:])[0]*1e3
            self.processed_results.loc[gene_id, 'r_i'] = np.mean(r['r_i'][f'circuit_{gene_id}'].loc[cutoff:])[0]*1e3


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


def terminate_at_threshold(t, y, *args):
    threshold = 1000.0
    return np.sqrt(np.mean(y**2)) - threshold


if __name__ == "__main__":
    cgs_worker = ExtendedWorker()
    cgs_worker.worker_init()
    #cgs_worker.worker_init(
    #    config_file="/nobackup/spanien1/rgast/PycharmProjects/BrainNetworks/BasalGanglia/results/Config/DefaultConfig_0.yaml",
    #    subgrid="/nobackup/spanien1/rgast/PycharmProjects/BrainNetworks/BasalGanglia/results/Grids/Subgrids/DefaultGrid_102/animals/animals_Subgrid_0.h5",
    #    result_file="~/my_result.h5",
    #    build_dir=os.getcwd()
    #)
