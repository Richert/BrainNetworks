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
        no_oscillations = [True, True, True, True, False, True]
        param_grid, invalid_params = eval_params(param_grid)
        conditions = [{},  # healthy control
                      {'k_ie': 0.0},  # STN blockade
                      {'k_ii': 0.0, 'k_str': 0.0},  # GABAA blockade in GPe
                      {'k_ie': 0.0, 'k_ii': 0.0, 'k_str': 0.0},
                      # STN blockade and GABAA blockade in GPe
                      {'k_ie': 0.0, 'eta_tha': 0.0},  # AMPA + NMDA blocker in GPe
                      {'k_ei': 0.0},  # GABAA antagonist in STN
                      ]
        chunk_size = [
            50,   # carpenters
            50,   # osttimor
            50,   # spanien
            100,  # animals
            20,   # kongo
            20,   # uganda
            20,   # tschad
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
                worker_kwargs={'no_oscillations': no_oscillations, 'targets': target},
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
    diff = np.asarray([0.0 if np.isnan(t_tmp) else y_tmp - t_tmp for y_tmp, t_tmp in zip(y, t)])
    return np.sqrt(np.mean(diff**2))


def eval_params(params):
    valid_params = []
    invalid_params = []
    for i, gene_id in enumerate(params.index):

        # check validity conditions
        valid = True
        if params.loc[gene_id, 'k_ee'] > 0.3*params.loc[gene_id, 'k_ie']:
            valid = False
        if params.loc[gene_id, 'k_ii'] > 0.6*params.loc[gene_id, 'k_ei']:
            valid = False
        if params.loc[gene_id, 'k_ie'] > 4.0*params.loc[gene_id, 'k_ei']:
            valid = False
        if params.loc[gene_id, 'k_ie'] < 0.25*params.loc[gene_id, 'k_ei']:
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
        'k_ee': {'min': 0, 'max': 10, 'size': 2, 'sigma': 0.4, 'loc': 3.0, 'scale': 0.5},
        'k_ei': {'min': 0, 'max': 120, 'size': 2, 'sigma': 0.8, 'loc': 30.0, 'scale': 10.0},
        'k_ie': {'min': 0, 'max': 120, 'size': 2, 'sigma': 0.8, 'loc': 30.0, 'scale': 10.0},
        'k_ii': {'min': 0, 'max': 120, 'size': 2, 'sigma': 0.8, 'loc': 15.0, 'scale': 5.0},
        'k_str': {'min': 0, 'max': 100, 'size': 2, 'sigma': 0.4, 'loc': 50.0, 'scale': 10.0},
        'eta_e': {'min': -30, 'max': 30, 'size': 2, 'sigma': 0.4, 'loc': 0.0, 'scale': 10.0},
        'eta_i': {'min': -30, 'max': 30, 'size': 2, 'sigma': 0.4, 'loc': 0.0, 'scale': 10.0},
        'eta_tha': {'min': 0, 'max': 30, 'size': 2, 'sigma': 0.4, 'loc': 10.0, 'scale': 5.0},
        'g_ee': {'min': 0, 'max': 1, 'size': 1, 'sigma': 0.05, 'loc': 0.5, 'scale': 0.1},
        'g_ie': {'min': 0, 'max': 1, 'size': 1, 'sigma': 0.1, 'loc': 0.5, 'scale': 0.1},
        'g_ei': {'min': 0, 'max': 1, 'size': 1, 'sigma': 0.1, 'loc': 0.5, 'scale': 0.1},
        'g_ii': {'min': 0, 'max': 1, 'size': 1, 'sigma': 0.1, 'loc': 0.5, 'scale': 0.1},
        'a_ee': {'min': 0, 'max': 10, 'size': 1, 'sigma': 0.2, 'loc': 1.0, 'scale': 0.2},
        'a_ei': {'min': 0, 'max': 10, 'size': 1, 'sigma': 0.2, 'loc': 1.0, 'scale': 0.2},
        'a_ie': {'min': 0, 'max': 10, 'size': 1, 'sigma': 0.2, 'loc': 1.0, 'scale': 0.2},
        'a_ii': {'min': 0, 'max': 10, 'size': 1, 'sigma': 0.2, 'loc': 1.0, 'scale': 0.2},
    }

    param_map = {
        'k_ee': {'vars': ['qif_stn/k_ee'], 'nodes': ['stn']},
        'k_ei': {'vars': ['qif_stn/k_ei'], 'nodes': ['stn']},
        'k_ie': {'vars': ['qif_gpe/k_ie'], 'nodes': ['gpe']},
        'k_ii': {'vars': ['qif_gpe/k_ii'], 'nodes': ['gpe']},
        'eta_e': {'vars': ['qif_stn/eta_e'], 'nodes': ['stn']},
        'eta_i': {'vars': ['qif_gpe/eta_i'], 'nodes': ['gpe']},
        'k_str': {'vars': ['qif_gpe/k_str'], 'nodes': ['gpe']},
        'eta_tha': {'vars': ['qif_gpe/eta_tha'], 'nodes': ['gpe']},
        'a_ee': {'vars': ['qif_stn/g_e'], 'nodes': ['stn']},
        'a_ei': {'vars': ['qif_stn/g_i'], 'nodes': ['stn']},
        'a_ie': {'vars': ['qif_gpe/g_e'], 'nodes': ['gpe']},
        'a_ii': {'vars': ['qif_gpe/g_i'], 'nodes': ['gpe']},
        'g_ee': {'vars': ['qif_stn/a_e'], 'nodes': ['stn']},
        'g_ei': {'vars': ['qif_stn/a_i'], 'nodes': ['stn']},
        'g_ie': {'vars': ['qif_gpe/a_e'], 'nodes': ['gpe']},
        'g_ii': {'vars': ['qif_gpe/a_i'], 'nodes': ['gpe']},
    }

    T = 5000.
    dt = 1e-2
    dts = 1e-1
    compute_dir = f"{os.getcwd()}/stn_gpe_healthy_opt"

    ga = CustomGOA(fitness_measure=fitness,
                   gs_config={
                       'circuit_template': f"{os.getcwd()}/config/stn_gpe/stn_gpe_syns",
                       'permute_grid': True,
                       'param_map': param_map,
                       'simulation_time': T,
                       'step_size': dt,
                       'sampling_step_size': dts,
                       'inputs': {},
                       'outputs': {'r_e': "stn/qif_stn/R_e", 'r_i': 'gpe/qif_gpe/R_i'},
                       'init_kwargs': {'backend': 'numpy', 'solver': 'scipy', 'step_size': dt},
                   },
                   cgs_config={'nodes': [
                                         'carpenters',
                                         'osttimor',
                                         'spanien',
                                         'animals',
                                         'kongo',
                                         'uganda',
                                         'tschad'
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
        target=[[20, 60],   # healthy control
                [np.nan, 40],  # stn blockade
                [np.nan, 90],  # gabaa blockade in GPe
                [np.nan, 60],  # STN blockade and GABAA blockade in GPe
                [np.nan, 17],  # blockade of all excitatory inputs to GPe
                [35, 80],  # GABAA antagonist in STN
                ],
        max_iter=100,
        min_fit=0.5,
        n_winners=10,
        n_parent_pairs=200,
        n_new=46,
        sigma_adapt=0.015,
        candidate_save=f'{compute_dir}/GeneticCGSCandidate.h5',
        drop_save=drop_save_dir,
        new_pop_on_drop=True,
        pop_save=f'{compute_dir}/Results/pop_summary'
    )

    winner.to_hdf(f'{compute_dir}/PopulationDrops/winner.h5', key='data')
