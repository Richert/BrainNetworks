# my_cgs_worker.py
from pyrates.utility.grid_search import ClusterWorkerTemplate
import os
from pandas import DataFrame, concat
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
        results, gene_ids = [], []
        param_grid_tmp = DataFrame.from_dict({key: param_grid[key] for key in model_vars})
        n = param_grid.shape[0]
        for i, c_dict in enumerate(conditions):
            for key in model_vars:
                if key in c_dict and type(c_dict[key]) is float:
                    c_dict[key] = np.zeros((param_grid.shape[0],)) + c_dict[key]
                else:
                    c_dict[key] = param_grid[key]
            param_grid_new = DataFrame.from_dict(c_dict)
            param_grid_new.index = np.arange(i*n, (i+1)*n)
            param_grid_tmp = concat((param_grid_tmp, param_grid_new), axis=0)
            gene_ids.append((i*n, (i+1)*n))
        r, self.result_map, t = grid_search(*args, param_grid=param_grid_tmp, **kwargs_tmp)
        new_labels = [f'circuit_{i}' for i in range(0, n)]
        for start, stop in gene_ids:
            targets = [f'circuit_{i}' for i in range(start, stop)]
            labels = list(r.columns.get_level_values(level=1))
            idx = [labels.index(t) for t in targets]
            r_tmp = r.iloc[:, idx]
            r_tmp.columns.set_levels(new_labels, level=1, inplace=True)
            results.append(r_tmp)
        self.results = results
        return t

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
            self.processed_results.loc[gene_id, 'fitness'] = 0.7*dist1+0.3*dist2
            self.processed_results.loc[gene_id, 'freq'] = freq[-1][idx]
            self.processed_results.loc[gene_id, 'pow'] = pow[-1][0][idx]
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


if __name__ == "__main__":
    cgs_worker = ExtendedWorker()
    cgs_worker.worker_init()
    #cgs_worker.worker_init(
    #    config_file="/nobackup/spanien1/rgast/PycharmProjects/BrainNetworks/BasalGanglia/results/Config/DefaultConfig_0.yaml",
    #    subgrid="/nobackup/spanien1/rgast/PycharmProjects/BrainNetworks/BasalGanglia/results/Grids/Subgrids/DefaultGrid_77/animals/animals_Subgrid_0.h5",
    #    result_file="~/my_result.h5",
    #    build_dir=os.getcwd()
    #)
