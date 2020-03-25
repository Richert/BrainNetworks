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
        conditions = kwargs_tmp.pop('conditions')
        param_grid = kwargs_tmp.pop('param_grid')
        results, gene_ids = [], param_grid.index
        for i, c_dict in enumerate(conditions):
            for key in param_grid:
                if '_pd' not in key:
                    if i < 6 or f"{key}_pd" not in param_grid:
                        param = param_grid[key]
                    else:
                        param = param_grid[key] + param_grid[f"{key}_pd"]
                    if key in c_dict:
                        c_dict[key] = param * c_dict[key]
                    else:
                        c_dict[key] = param
            param_grid_tmp = DataFrame.from_dict(c_dict)
            kwargs_gs = deepcopy(kwargs_tmp)
            r, self.result_map, sim_time = grid_search(*args, param_grid=param_grid_tmp, **kwargs_gs)
            r = r.droplevel(2, axis=1)
            results.append(r)
            self.results = results
        return sim_time

    def worker_postprocessing(self, **kwargs):
        kwargs_tmp = kwargs.copy()
        param_grid = kwargs_tmp.pop('param_grid')
        freq_targets = kwargs_tmp.pop('freq_targets')
        targets = kwargs_tmp.pop('targets')
        self.processed_results = DataFrame(data=None, columns=['fitness', 'r_e', 'r_p', 'r_a', 'r_s'])

        # calculate fitness
        for gene_id in param_grid.index:
            outputs, vars, = [], []
            for i, r in enumerate(self.results):
                r = r * 1e3
                r.index = r.index * 1e-3
                cutoff = r.index[-1]*0.8
                mean_re = np.mean(r['r_e'][f'circuit_{gene_id}'].loc[cutoff:])
                mean_rp = np.mean(r['r_p'][f'circuit_{gene_id}'].loc[cutoff:])
                mean_ra = np.mean(r['r_a'][f'circuit_{gene_id}'].loc[cutoff:])
                mean_rs = np.mean(r['r_s'][f'circuit_{gene_id}'].loc[cutoff:])
                var_re = np.var(r['r_e'][f'circuit_{gene_id}'].loc[cutoff:])
                var_rp = np.var(r['r_p'][f'circuit_{gene_id}'].loc[cutoff:])
                var_ra = np.var(r['r_a'][f'circuit_{gene_id}'].loc[cutoff:])
                var_rs = np.var(r['r_s'][f'circuit_{gene_id}'].loc[cutoff:])
                outputs.append([mean_re, mean_rp, mean_ra, mean_rs])
                vars.append(np.mean([var_re, var_rp, var_ra, var_rs]))

            for m in range(1, len(targets)):
                for n in range(len(targets[m])):
                    if targets[m][n] != np.nan:
                        targets[m][n] = outputs[0][n] * targets[m][n]
            dist1 = fitness(outputs, targets)
            dist2 = fitness(vars, freq_targets)
            self.processed_results.loc[gene_id, 'fitness'] = dist1+dist2
            self.processed_results.loc[gene_id, 'r_e'] = [rates[0] for rates in outputs]
            self.processed_results.loc[gene_id, 'r_p'] = [rates[1] for rates in outputs]
            self.processed_results.loc[gene_id, 'r_a'] = [rates[2] for rates in outputs]
            self.processed_results.loc[gene_id, 'r_s'] = [rates[3] for rates in outputs]


def fitness(y, t):
    y = np.asarray(y).flatten()
    t = np.asarray(t).flatten()
    diff = np.asarray([0.0 if np.isnan(t_tmp) else y_tmp - t_tmp for y_tmp, t_tmp in zip(y, t)]).flatten()
    t[np.isnan(t)] = 1.0
    t[t == 0] = 1.0
    weights = 1/np.abs(t)
    return weights @ np.abs(diff)


if __name__ == "__main__":
    cgs_worker = ExtendedWorker()
    cgs_worker.worker_init()
    #cgs_worker.worker_init(
    #    config_file="/nobackup/spanien1/rgast/PycharmProjects/BrainNetworks/BasalGanglia/stn_gpe_combined_opt1/Config/DefaultConfig_0.yaml",
    #    subgrid="/nobackup/spanien1/rgast/PycharmProjects/BrainNetworks/BasalGanglia/stn_gpe_combined_opt1/Grids/Subgrids/DefaultGrid_2/animals/animals_Subgrid_1.h5",
    #    result_file="~/my_result.h5",
    #    build_dir=os.getcwd()
    #)
