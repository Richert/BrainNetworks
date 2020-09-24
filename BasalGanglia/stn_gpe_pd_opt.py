import os
import warnings
import numpy as np
from pyrates.utility.genetic_algorithm import CGSGeneticAlgorithm
from pandas import DataFrame, read_hdf
from copy import deepcopy


class CustomGOA(CGSGeneticAlgorithm):

    def eval_fitness(self, target: list, **kwargs):

        # define simulation conditions
        worker_file = self.cgs_config['worker_file'] if 'worker_file' in self.cgs_config else None
        param_grid = self.pop.drop(['fitness', 'sigma', 'results'], axis=1)
        result_vars = ['r_e', 'r_i']
        freq_targets = [0.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        param_grid, invalid_params = eval_params(param_grid)
        conditions = [{},  # pd control
                      {'k_ie': 0.2, 'eta_tha': 0.2},  # AMPA blockade in GPe
                      {'k_ii': 0.2, 'k_str': 0.2},  # GABAA blockade in GPe
                      {'k_ee': 0.2, 'eta_ee': 0.2},  # AMPA blockade in STN
                      {'k_ei': 0.0},  # GPe blockade
                      ]
        chunk_size = [
            #100,   # carpenters
            #150,   # osttimor
            150,   # spanien
            150,   # animals
            50,   # kongo
        ]

        # perform simulations
        if len(param_grid) > 0:
            self.gs_config['init_kwargs'].update(kwargs)
            res_file = self.cgs.run(
                circuit_template=self.gs_config['circuit_template'],
                param_grid=deepcopy(param_grid),
                param_map=self.gs_config['param_map'],
                simulation_time=self.gs_config['simulation_time'],
                dt=self.gs_config['step_size'],
                inputs=self.gs_config['inputs'],
                outputs=self.gs_config['outputs'],
                sampling_step_size=self.gs_config['sampling_step_size'],
                permute=False,
                chunk_size=chunk_size,
                worker_file=worker_file,
                worker_env=self.cgs_config['worker_env'],
                gs_kwargs={'init_kwargs': self.gs_config['init_kwargs'], 'conditions': conditions},
                worker_kwargs={'freq_targets': freq_targets, 'y': target},
                result_concat_axis=0)
            results_tmp = read_hdf(res_file, key=f'Results/results')

            # calculate fitness
            for gene_id in param_grid.index:
                self.pop.at[gene_id, 'fitness'] = 1.0 / results_tmp.at[gene_id, 'fitness']
                self.pop.at[gene_id, 'results'] = [results_tmp.at[gene_id, v] for v in result_vars]

        # set fitness of invalid parametrizations
        for gene_id in invalid_params.index:
            self.pop.at[gene_id, 'fitness'] = 0.0
            self.pop.at[gene_id, 'results'] = [0. for _ in result_vars]


def fitness(y, t):
    y = np.asarray(y).flatten()
    t = np.asarray(t).flatten()
    diff = np.asarray([0.0 if np.isnan(t_tmp) else y_tmp - t_tmp for y_tmp, t_tmp in zip(y, t)]).flatten()
    t[np.isnan(t)] = 1.0
    t[t == 0] = 1.0
    weights = 1/np.abs(t)
    return weights @ np.abs(diff)


def eval_params(params):
    valid_params = []
    invalid_params = []
    for i, gene_id in enumerate(params.index):

        # check validity conditions
        valid = True
        if params.loc[gene_id, 'k_ee'] > 0.3*params.loc[gene_id, 'k_ie']:
            valid = False
        if params.loc[gene_id, 'k_ie'] > 10.0*params.loc[gene_id, 'k_ei']:
            valid = False
        if params.loc[gene_id, 'k_ie'] < 0.1*params.loc[gene_id, 'k_ei']:
            valid = False

        # add parametrization to valid or invalid parameter sets
        if valid:
            valid_params.append(i)
        else:
            invalid_params.append(i)

    valid_df = params.iloc[valid_params, :]
    valid_df.index = valid_params

    invalid_df = params.iloc[invalid_params, :]
    invalid_df.index = invalid_params

    return valid_df, invalid_df


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    pop_genes = {
        'k_ee': {'min': 0, 'max': 30, 'size': 2, 'sigma': 0.1, 'loc': 5.0, 'scale': 0.5},
        'k_ei': {'min': 0, 'max': 200, 'size': 2, 'sigma': 1.0, 'loc': 100.0, 'scale': 20.0},
        'k_ie': {'min': 0, 'max': 200, 'size': 2, 'sigma': 1.0, 'loc': 100.0, 'scale': 20.0},
        'k_ii': {'min': 0, 'max': 150, 'size': 2, 'sigma': 1.0, 'loc': 50.0, 'scale': 10.0},
        'k_str': {'min': 0, 'max': 800, 'size': 2, 'sigma': 5.0, 'loc': 100.0, 'scale': 20.0},
        'eta_e': {'min': -20, 'max': 40, 'size': 1, 'sigma': 1.0, 'loc': 0.0, 'scale': 5.0},
        'eta_i': {'min': -20, 'max': 40, 'size': 2, 'sigma': 1.0, 'loc': 10.0, 'scale': 5.0},
        'eta_tha': {'min': 0, 'max': 60, 'size': 2, 'sigma': 1.0, 'loc': 20.0, 'scale': 5.0},
        'eta_ee': {'min': 0.0, 'max': 60, 'size': 2, 'sigma': 1.0, 'loc': 20.0, 'scale': 5.0},
        'delta_e': {'min': 0.5, 'max': 15.0, 'size': 2, 'sigma': 0.05, 'loc': 2.0, 'scale': 0.2},
        'delta_i': {'min': 0.5, 'max': 15.0, 'size': 2, 'sigma': 0.05, 'loc': 4.0, 'scale': 0.4}
    }

    param_map = {
        'k_ee': {'vars': ['stn_basic/k_ee'], 'nodes': ['stn']},
        'k_ei': {'vars': ['stn_basic/k_ei'], 'nodes': ['stn']},
        'k_ie': {'vars': ['gpe_basic/k_ie'], 'nodes': ['gpe']},
        'k_ii': {'vars': ['gpe_basic/k_ii'], 'nodes': ['gpe']},
        'eta_e': {'vars': ['stn_basic/eta_e'], 'nodes': ['stn']},
        'eta_ee': {'vars': ['stn_basic/eta_ee'], 'nodes': ['stn']},
        'eta_i': {'vars': ['gpe_basic/eta_i'], 'nodes': ['gpe']},
        'k_str': {'vars': ['gpe_basic/k_str'], 'nodes': ['gpe']},
        'eta_tha': {'vars': ['gpe_basic/eta_tha'], 'nodes': ['gpe']},
        'delta_e': {'vars': ['stn_basic/delta_e'], 'nodes': ['stn']},
        'delta_i': {'vars': ['gpe_basic/delta_i'], 'nodes': ['gpe']}
    }

    T = 2000.
    dt = 1e-2
    dts = 1e-1

    # perform genetic optimization

    compute_dir = f"{os.getcwd()}/stn_gpe_healthy_opt1"

    ga = CustomGOA(fitness_measure=fitness,
                   gs_config={
                       'circuit_template': f"{os.getcwd()}/config/stn_gpe/stn_gpe_basic",
                       'permute_grid': True,
                       'param_map': param_map,
                       'simulation_time': T,
                       'step_size': dt,
                       'sampling_step_size': dts,
                       'inputs': {},
                       'outputs': {'r_e': "stn/stn_basic/R_e", 'r_i': 'gpe/gpe_basic/R_i'},
                       'init_kwargs': {'backend': 'numpy', 'solver': 'scipy', 'step_size': dt},
                   },
                   cgs_config={'nodes': [
                                         #'carpenters',
                                         #'osttimor',
                                         'spanien',
                                         'animals',
                                         'kongo',
                                         #'tschad'
                                         ],
                               'compute_dir': compute_dir,
                               'worker_file': f'{os.getcwd()}/stn_gpe_healthy_worker.py',
                               'worker_env': "/nobackup/spanien1/rgast/anaconda3/envs/pyrates_test/bin/python3",
                               })

    drop_save_dir = f'{compute_dir}/PopulationDrops/'
    os.makedirs(drop_save_dir, exist_ok=True)

    winner = ga.run(
        initial_gene_pool=pop_genes,
        gene_sampling_func=np.random.normal,
        new_member_sampling_func=np.random.uniform,
        target=[[30, 40, 14, 14],   # healthy control
                [np.nan, 20, np.nan, 0.0],  # ampa blockade in GPe
                [np.nan, 90, np.nan, 14],  # gabaa blockade in GPe
                [30, np.nan, 0.0, np.nan],  # AMPA blockade in STN
                [50, np.nan, 0.0, np.nan],  # GPe blockade
                ],
        max_iter=100,
        enforce_max_iter=True,
        min_fit=2.0,
        n_winners=10,
        n_parent_pairs=800,
        n_new=214,
        sigma_adapt=0.05,
        candidate_save=f'{compute_dir}/GeneticCGSCandidate.h5',
        drop_save=drop_save_dir,
        new_pop_on_drop=True,
        pop_save=f'{drop_save_dir}/pop_summary'
    )

    #winner.to_hdf(f'{drop_save_dir}/winner.h5', key='data')
