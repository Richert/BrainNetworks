# my_cgs_worker.py
from pyrates.utility.grid_search import ClusterWorkerTemplate
import os
from pandas import DataFrame
from pyrates.utility import grid_search, welch
import numpy as np
from copy import deepcopy
from scipy.ndimage.filters import gaussian_filter1d


class MinimalWorker(ClusterWorkerTemplate):
    def worker_postprocessing(self, **kwargs):
        self.processed_results = DataFrame(data=None, columns=self.results.columns)
        for idx, data in self.results.iteritems():
            self.processed_results.loc[:, idx] = data * 1e3
        self.processed_results.index = self.results.index * 1e-3


class ExtendedWorker(MinimalWorker):
    def worker_gs(self, *args, **kwargs):

        # get parameters
        kwargs_tmp = deepcopy(kwargs)
        conditions = kwargs_tmp.pop('conditions')
        param_grid = kwargs_tmp.pop('param_grid')
        param_scalings = kwargs_tmp.pop('param_scalings')

        # simulate autonomous behavior in different conditions
        results, gene_ids = [], param_grid.index
        for c_dict in deepcopy(conditions):
            for key in param_grid:
                if key in c_dict:
                    c_dict[key] = deepcopy(param_grid[key]) * c_dict[key]
                elif key in param_grid:
                    c_dict[key] = deepcopy(param_grid[key])
            for key, key_tmp, power in param_scalings:
                c_dict[key] = c_dict[key] * c_dict[key_tmp]**power
            param_grid_tmp = DataFrame.from_dict(c_dict)
            r, self.result_map, sim_time = grid_search(*args, param_grid=param_grid_tmp, **deepcopy(kwargs_tmp))
            r = r.droplevel(2, axis=1)
            results.append(r)

        self.results = results
        return sim_time

    def worker_postprocessing(self, **kwargs):
        kwargs_tmp = kwargs.copy()
        param_grid = kwargs_tmp.pop('param_grid')
        freq_targets = kwargs_tmp.pop('freq_targets')
        targets = kwargs_tmp.pop('targets')
        self.processed_results = DataFrame(data=None, columns=['fitness', 'r_e', 'r_i', 'r_a'])

        # calculate fitness
        for gene_id in param_grid.index:
            outputs, vars = [], []
            for i, r in enumerate(self.results):
                r = r * 1e3
                r.index = r.index * 1e-3
                cutoff = r.index[-1]*0.5
                cutoff2 = r.index[-1]*0.25
                func = np.mean #if i < len(self.results)-1 else np.max
                mean_re = func(r['r_e'][f'circuit_{gene_id}'].loc[cutoff:])
                mean_ri = func(r['r_i'][f'circuit_{gene_id}'].loc[cutoff:])
                mean_ra = func(r['r_a'][f'circuit_{gene_id}'].loc[cutoff:])
                outputs.append([mean_re, mean_ri, mean_ra])
                vars.append(np.var(r['r_i'][f'circuit_{gene_id}'].loc[cutoff:]) if freq_targets[i] == 0.0 else
                            1/np.var(r['r_i'][f'circuit_{gene_id}'].loc[cutoff2:]))
                freq_targets[i] = 0.0

            dist1 = fitness(outputs, targets)
            dist2 = fitness(vars, freq_targets)

            self.processed_results.loc[gene_id, 'fitness'] = 1/(dist1+dist2)
            self.processed_results.loc[gene_id, 'r_e'] = [rates[0] for rates in outputs]
            self.processed_results.loc[gene_id, 'r_i'] = [rates[1] for rates in outputs]
            self.processed_results.loc[gene_id, 'r_a'] = [rates[2] for rates in outputs]


def fitness(y, t, squared=True):
    y = np.asarray(y).flatten()
    t = np.asarray(t).flatten()
    diff = np.asarray([0.0 if np.isnan(t_tmp) else y_tmp - t_tmp for y_tmp, t_tmp in zip(y, t)]).flatten()
    diff = diff**2 if squared else np.abs(diff)
    t[np.isnan(t)] = 1.0
    t[t == 0] = 1.0
    weights = 1/np.abs(t)
    return weights @ np.abs(diff)


if __name__ == "__main__":
    cgs_worker = ExtendedWorker()
    cgs_worker.worker_init()
    #cgs_worker.worker_init(
    #    config_file="/data/u_rgast_software/PycharmProjects/BrainNetworks/BasalGanglia/stn_gpe_healthy_opt/Config/DefaultConfig_0.yaml",
    #    subgrid="/data/u_rgast_software/PycharmProjects/BrainNetworks/BasalGanglia/stn_gpe_healthy_opt/Grids/Subgrids/DefaultGrid_38/spanien/spanien_Subgrid_0.h5",
    #    result_file="~/my_result.h5",
    #    build_dir=os.getcwd()
    #
