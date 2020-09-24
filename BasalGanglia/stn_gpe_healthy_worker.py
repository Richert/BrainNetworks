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
        T = kwargs_tmp['simulation_time']
        dt = kwargs_tmp['step_size']

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

        # simulate behavior under extrinsic forcing
        sim_steps = int(np.round(T / dt))
        stim_offset = int(np.round(T * 0.5 / dt))
        stim_dur = int(np.round(1.0 / dt))
        stim_delayed = int(np.round((T * 0.5 + 14.0) / dt))
        stim_amp = 9.0
        stim_var = 100.0
        ctx = np.zeros((sim_steps, 1))
        ctx[stim_offset:stim_offset + stim_dur, 0] = stim_amp
        ctx = gaussian_filter1d(ctx, stim_var, axis=0)
        stria = np.zeros((sim_steps, 1))
        stria[stim_delayed:stim_delayed + stim_dur, 0] = stim_amp
        stria = gaussian_filter1d(stria, stim_var * 10.0, axis=0)
        for key, key_tmp, power in param_scalings:
            param_grid[key] = np.asarray(param_grid[key]) * np.asarray(param_grid[key_tmp]) ** power
        kwargs_tmp.pop('inputs')
        r, self.result_map, sim_time = grid_search(*args, param_grid=param_grid,
                                                   inputs={'stn/stn_op/ctx': ctx, 'str/str_dummy_op/I': stria},
                                                   **kwargs_tmp)
        results.append(r)

        self.results = results
        return sim_time

    def worker_postprocessing(self, **kwargs):
        kwargs_tmp = kwargs.copy()
        param_grid = kwargs_tmp.pop('param_grid')
        freq_targets = kwargs_tmp.pop('freq_targets')
        targets = kwargs_tmp.pop('y')
        T = kwargs_tmp['T']
        self.processed_results = DataFrame(data=None, columns=['fitness', 'r_e', 'r_i', 'r_a'])

        # calculate fitness
        for gene_id in param_grid.index:
            outputs, vars = [], []
            for i, r in enumerate(self.results[:-1]):
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

            stim_t = T * 0.5
            times = [stim_t,
                     stim_t + 5.0,
                     stim_t + 8.0,
                     stim_t + 20.0,
                     stim_t + 25.0,
                     stim_t + 35.0,
                     stim_t + 60.0
                     ]
            targets_t = [
                [20.0, 60.0],  # steady-state (t = stim_t)
                [150.0, 60.0],  # first peak stn (t = stim_t + 5.0)
                [np.nan, 400.0],  # first peak gpe (t = stim_t + 8.0))
                [200.0, 2.0],  # second stn peak (t = sim_t + 20.0)
                [np.nan, 400.0],  # second gpe peak (t = sim_t + 25.0)
                [10.0, 200.0],  # fall of of second gpe peak (t = sim_t + 35.0)
                [20.0, 60.0],  # steady-sate (t = stim_t + 60.0)
            ]
            r = self.results[-1]
            results_t = [[r.loc[t, 'r_e'], r.loc[t, 'r_i']] for t in times]
            dist3 = fitness(results_t, targets_t, squared=False)
            self.processed_results.loc[gene_id, 'fitness'] = dist1+dist2+dist3
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
    #)
