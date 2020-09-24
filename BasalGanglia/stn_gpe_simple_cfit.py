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
        param_grid, invalid_params = eval_params(param_grid)
        chunk_size = [
            300,   # carpenters
            300,   # osttimor
            200,   # spanien
            300,  # animals
            100,   # kongo
            100,   # uganda
            #100,   # tschad
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
                gs_kwargs={'init_kwargs': self.gs_config['init_kwargs']},
                worker_kwargs={'y': target},
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
    t = np.asarray(t)
    weights = t/sum(t)
    y = np.asarray(y).flatten()
    t = np.asarray(t).flatten()
    diff = np.asarray([0.0 if np.isnan(t_tmp) else y_tmp - t_tmp for y_tmp, t_tmp in zip(y, t)])
    return np.sqrt(weights @ diff**2)


def eval_params(params):
    valid_params = []
    invalid_params = []
    for i, gene_id in enumerate(params.index):

        # check validity conditions
        valid = True
        if params.loc[gene_id, 'J_ee'] > 0.3*params.loc[gene_id, 'J_ie']:
            valid = False
        if params.loc[gene_id, 'J_ie'] > 10.0*params.loc[gene_id, 'J_ei']:
            valid = False
        if params.loc[gene_id, 'J_ie'] < 0.1*params.loc[gene_id, 'J_ei']:
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
        'J_ee': {'min': 0, 'max': 10, 'size': 4, 'sigma': 0.1, 'loc': 2.0, 'scale': 0.5},
        'J_ei': {'min': 0, 'max': 120, 'size': 4, 'sigma': 1.0, 'loc': 50.0, 'scale': 10.0},
        'J_ie': {'min': 0, 'max': 120, 'size': 4, 'sigma': 1.0, 'loc': 50.0, 'scale': 10.0},
        'J_ii': {'min': 0, 'max': 120, 'size': 4, 'sigma': 1.0, 'loc': 20.0, 'scale': 10.0},
        'eta_e': {'min': -20, 'max': 30, 'size': 4, 'sigma': 1.0, 'loc': 0.0, 'scale': 10.0},
        'eta_i': {'min': -20, 'max': 30, 'size': 4, 'sigma': 1.0, 'loc': 20.0, 'scale': 10.0},
    }

    param_map = {
        'J_ee': {'vars': ['qif_simple/J_ee'], 'nodes': ['stn_gpe']},
        'J_ei': {'vars': ['qif_simple/J_ei'], 'nodes': ['stn_gpe']},
        'J_ie': {'vars': ['qif_simple/J_ie'], 'nodes': ['stn_gpe']},
        'J_ii': {'vars': ['qif_simple/J_ii'], 'nodes': ['stn_gpe']},
        'eta_e': {'vars': ['qif_simple/eta_e'], 'nodes': ['stn_gpe']},
        'eta_i': {'vars': ['qif_simple/eta_i'], 'nodes': ['stn_gpe']},
    }

    T = 5000.
    dt = 1e-2
    dts = 1e-1
    compute_dir = f"{os.getcwd()}/stn_gpe_simple_opt"

    ga = CustomGOA(fitness_measure=fitness,
                   gs_config={
                       'circuit_template': f"{os.getcwd()}/config/stn_gpe/stn_gpe_reduced",
                       'permute_grid': True,
                       'param_map': param_map,
                       'simulation_time': T,
                       'step_size': dt,
                       'sampling_step_size': dts,
                       'inputs': {},
                       'outputs': {'r_e': "stn_gpe/qif_simple/R_e", 'r_i': 'stn_gpe/qif_simple/R_i'},
                       'init_kwargs': {'backend': 'numpy', 'solver': 'scipy', 'step_size': dt},
                   },
                   cgs_config={'nodes': [
                                         'carpenters',
                                         'osttimor',
                                         'spanien',
                                         'animals',
                                         'kongo',
                                         'uganda',
                                         #'tschad'
                                         ],
                               'compute_dir': compute_dir,
                               'worker_file': f'{os.getcwd()}/stn_gpe_simple_cfit_worker.py',
                               'worker_env': "/nobackup/spanien1/rgast/anaconda3/envs/pyrates_test/bin/python3",
                               })

    drop_save_dir = f'{compute_dir}/PopulationDrops/'
    os.makedirs(drop_save_dir, exist_ok=True)

    winner = ga.run(
        initial_gene_pool=pop_genes,
        gene_sampling_func=np.random.normal,
        new_member_sampling_func=np.random.uniform,
        target=[20.0, 60.0],
        max_iter=100,
        min_fit=1.0,
        n_winners=20,
        n_parent_pairs=3600,
        n_new=476,
        sigma_adapt=0.015,
        candidate_save=f'{compute_dir}/GeneticCGSCandidate.h5',
        drop_save=drop_save_dir,
        new_pop_on_drop=True,
        pop_save=f'{compute_dir}/Results/pop_summary'
    )

    winner.to_hdf(f'{compute_dir}/PopulationDrops/winner.h5', key='data')
