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
        targets = kwargs_tmp.pop('y')
        self.processed_results = DataFrame(data=None, columns=['fitness', 'r_e', 'r_i'])

        # calculate fitness
        for gene_id in param_grid.index:
            r = self.results
            r = r * 1e3
            r.index = r.index * 1e-3
            cutoff = r.index[-1]*0.5
            mean_re = np.mean(r['r_e'][f'circuit_{gene_id}'].loc[cutoff:])
            mean_ri = np.mean(r['r_i'][f'circuit_{gene_id}'].loc[cutoff:])
            outputs = [mean_re, mean_ri]

            dist1 = fitness(outputs, targets)
            dist2 = np.var(r['r_i'][f'circuit_{gene_id}'].loc[cutoff:])
            r = self.results
            self.processed_results.loc[gene_id, 'fitness'] = dist1+dist2
            self.processed_results.loc[gene_id, 'r_e'] = np.mean(r['r_e'][f'circuit_{gene_id}'].loc[cutoff:])*1e3
            self.processed_results.loc[gene_id, 'r_i'] = np.mean(r['r_i'][f'circuit_{gene_id}'].loc[cutoff:])*1e3


def fitness(y, t):
    t = np.asarray(t)
    weights = t/sum(t)
    y = np.asarray(y).flatten()
    t = np.asarray(t).flatten()
    diff = np.asarray([0.0 if np.isnan(t_tmp) else y_tmp - t_tmp for y_tmp, t_tmp in zip(y, t)])
    return np.sqrt(weights @ diff**2)


if __name__ == "__main__":
    cgs_worker = ExtendedWorker()
    cgs_worker.worker_init()
    #cgs_worker.worker_init(
    #    config_file="/nobackup/spanien1/rgast/PycharmProjects/BrainNetworks/BasalGanglia/stn_gpe_simple_opt/Config/DefaultConfig_0.yaml",
    #    subgrid="/nobackup/spanien1/rgast/PycharmProjects/BrainNetworks/BasalGanglia/stn_gpe_simple_opt/Grids/Subgrids/DefaultGrid_43/spanien/spanien_Subgrid_0.h5",
    #    result_file="~/my_result.h5",
    #    build_dir=os.getcwd()
    #)
