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
        result_vars = ['r_e', 'r_i', 'r_a']
        freq_targets = [0.0, np.nan, np.nan, np.nan, np.nan]
        #param_grid, invalid_params = eval_params(param_grid)
        conditions = [{},  # healthy control
                      {'k_pe': 0.2, 'k_ae': 0.2, 'eta_tha': 0.2},  # AMPA blockade in GPe
                      {'k_pe': 0.2, 'k_pp': 0.2, 'k_pa': 0.2, 'k_ae': 0.2, 'k_aa': 0.2, 'k_ap': 0.2,
                       'eta_str': 0.2, 'eta_tha': 0.2},  # AMPA blockade and GABAA blockade in GPe
                      {'k_pp': 0.2, 'k_pa': 0.2, 'k_aa': 0.2, 'k_ap': 0.2, 'eta_str': 0.2},  # GABAA blockade in GPe
                      {'k_pe': 0.0, 'k_ae': 0.2},  # STN blockade
                      {'k_ep': 0.2}  # GABAA blocker in STN
                      ]
        param_scalings = [
            #('delta_e', 'tau_e', 2.0),
            #('delta_i', 'tau_i', 2.0),
            #('k_ee', 'delta_e', 0.5),
            #('k_ei', 'delta_e', 0.5),
            #('k_ie', 'delta_i', 0.5),
            #('k_ii', 'delta_i', 0.5),
            #('eta_e', 'delta_e', 1.0),
            #('eta_i', 'delta_i', 1.0),
            ]
        chunk_size = [
            50,  # carpenters
            100,  # osttimor
            50,  # spanien
            100,  # animals
            50,  # kongo
            50,  # tschad
            #100,  # uganda
            #50,  # tiber
            #50,  # giraffe
            20,  # lech
            10,  # rilke
            10,  # dinkel
            #10,  # rosmarin
            #10,  # mosambik
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
                gs_kwargs={'init_kwargs': self.gs_config['init_kwargs'], 'conditions': conditions,
                           'param_scalings': param_scalings},
                worker_kwargs={'freq_targets': freq_targets, 'targets': target, 'time_lim': 1800.0},
                result_concat_axis=0)
            results_tmp = read_hdf(res_file, key=f'Results/results')

            # calculate fitness
            for gene_id in param_grid.index:
                self.pop.at[gene_id, 'fitness'] = 1.0 / results_tmp.at[gene_id, 'fitness']
                self.pop.at[gene_id, 'results'] = [results_tmp.at[gene_id, v] for v in result_vars]

        # set fitness of invalid parametrizations
        #for gene_id in invalid_params.index:
        #    self.pop.at[gene_id, 'fitness'] = 0.0
        #    self.pop.at[gene_id, 'results'] = [0. for _ in result_vars]


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
        if params.loc[gene_id, 'k_ee'] > 0.2*params.loc[gene_id, 'k_ie']:
            valid = False
        if params.loc[gene_id, 'k_ie'] > 5.0*params.loc[gene_id, 'k_ei']:
            valid = False
        if params.loc[gene_id, 'k_ie'] < 0.2*params.loc[gene_id, 'k_ei']:
            valid = False
        if params.loc[gene_id, 'k_ii'] > 0.8*params.loc[gene_id, 'k_ei']:
            valid = False
        if params.loc[gene_id, 'tau_e'] > params.loc[gene_id, 'tau_i']:
            valid = False
        if params.loc[gene_id, 'delta_e'] > params.loc[gene_id, 'delta_i']:
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

    pop_size = 512
    pop_genes = {
        'k_ee': {'min': 0, 'max': 50, 'size': pop_size, 'sigma': 0.1, 'loc': 1.0, 'scale': 0.5},
        'k_ae': {'min': 10, 'max': 400, 'size': pop_size, 'sigma': 0.5, 'loc': 20.0, 'scale': 2.0},
        'k_pe': {'min': 10, 'max': 400, 'size': pop_size, 'sigma': 0.5, 'loc': 20.0, 'scale': 2.0},
        'k_pp': {'min': 5, 'max': 200, 'size': pop_size, 'sigma': 0.5, 'loc': 10.0, 'scale': 1.0},
        'k_ep': {'min': 5, 'max': 200, 'size': pop_size, 'sigma': 0.5, 'loc': 20.0, 'scale': 2.0},
        'k_ap': {'min': 5, 'max': 200, 'size': pop_size, 'sigma': 0.5, 'loc': 10.0, 'scale': 1.0},
        'k_aa': {'min': 2, 'max': 100, 'size': pop_size, 'sigma': 0.5, 'loc': 10.0, 'scale': 1.0},
        'k_pa': {'min': 2, 'max': 100, 'size': pop_size, 'sigma': 0.5, 'loc': 10.0, 'scale': 1.0},
        'eta_e': {'min': -50, 'max': 50, 'size': pop_size, 'sigma': 0.5, 'loc': 0.5, 'scale': 1.0},
        'eta_p': {'min': -50, 'max': 50, 'size': pop_size, 'sigma': 0.5, 'loc': 0.5, 'scale': 1.0},
        'eta_a': {'min': -50, 'max': 50, 'size': pop_size, 'sigma': 0.5, 'loc': 0.5, 'scale': 1.0},
        'eta_tha': {'min': 0, 'max': 100, 'size': pop_size, 'sigma': 0.5, 'loc': 5.0, 'scale': 1.0},
        'eta_str': {'min': -100, 'max': 0, 'size': pop_size, 'sigma': 0.5, 'loc': -20.0, 'scale': 1.0},
        'delta_e': {'min': 0.5, 'max': 20.0, 'size': pop_size, 'sigma': 0.5, 'loc': 2.0, 'scale': 0.5},
        'delta_p': {'min': 0.5, 'max': 40, 'size': pop_size, 'sigma': 0.5, 'loc': 4.0, 'scale': 1.0},
        'delta_a': {'min': 1.0, 'max': 80, 'size': pop_size, 'sigma': 0.5, 'loc': 8.0, 'scale': 1.0},
        'tau_e': {'min': 12, 'max': 12, 'size': pop_size, 'sigma': 0.0, 'loc': 12.0, 'scale': 0.0},
        'tau_p': {'min': 24, 'max': 24, 'size': pop_size, 'sigma': 0.0, 'loc': 24.0, 'scale': 0.0},
        'tau_a': {'min': 20, 'max': 20, 'size': pop_size, 'sigma': 0.0, 'loc': 20.0, 'scale': 0.0},
    }

    param_map = {
        'k_ee': {'vars': ['weight'], 'edges': [('stn', 'stn')]},
        'k_ae': {'vars': ['weight'], 'edges': [('stn', 'gpe_a')]},
        'k_pe': {'vars': ['weight'], 'edges': [('stn', 'gpe_p')]},
        'k_pp': {'vars': ['weight'], 'edges': [('gpe_p', 'gpe_p')]},
        'k_ep': {'vars': ['weight'], 'edges': [('gpe_p', 'stn')]},
        'k_ap': {'vars': ['weight'], 'edges': [('gpe_p', 'gpe_a')]},
        'k_aa': {'vars': ['weight'], 'edges': [('gpe_a', 'gpe_a')]},
        'k_pa': {'vars': ['weight'], 'edges': [('gpe_a', 'gpe_p')]},
        'eta_e': {'vars': ['stn_op/eta_e'], 'nodes': ['stn']},
        'eta_p': {'vars': ['gpe_proto_op/eta_i'], 'nodes': ['gpe_p']},
        'eta_a': {'vars': ['gpe_arky_op/eta_a'], 'nodes': ['gpe_a']},
        'eta_tha': {'vars': ['gpe_proto_op/eta_tha', 'gpe_arky_op/eta_tha'], 'nodes': ['gpe_p', 'gpe_a']},
        'eta_str': {'vars': ['gpe_proto_op/eta_str', 'gpe_arky_op/eta_str'], 'nodes': ['gpe_p', 'gpe_a']},
        'delta_e': {'vars': ['stn_op/delta_e'], 'nodes': ['stn']},
        'delta_p': {'vars': ['gpe_proto_op/delta_i'], 'nodes': ['gpe_p']},
        'delta_a': {'vars': ['gpe_arky_op/delta_a'], 'nodes': ['gpe_a']},
        'tau_e': {'vars': ['stn_op/tau_e'], 'nodes': ['stn']},
        'tau_p': {'vars': ['gpe_proto_op/tau_i'], 'nodes': ['gpe_p']},
        'tau_a': {'vars': ['gpe_arky_op/tau_a'], 'nodes': ['gpe_a']},
    }

    T = 2000.
    dt = 1e-2
    dts = 1e-1

    # perform genetic optimization
    compute_dir = f"{os.getcwd()}/stn_gpe_healthy_opt"

    ga = CustomGOA(fitness_measure=fitness,
                   gs_config={
                       'circuit_template': f"{os.getcwd()}/config/stn_gpe/stn_gpe",
                       'permute_grid': True,
                       'param_map': param_map,
                       'simulation_time': T,
                       'step_size': dt,
                       'sampling_step_size': dts,
                       'inputs': {},
                       'outputs': {'r_e': "stn/stn_op/R_e",
                                   'r_i': 'gpe_p/gpe_proto_op/R_i',
                                   'r_a': 'gpe_a/gpe_arky_op/R_a'},
                       'init_kwargs': {'backend': 'numpy', 'solver': 'scipy', 'step_size': dt},
                   },
                   cgs_config={'nodes': [
                                           'carpenters',
                                           'osttimor',
                                           'spanien',
                                           'animals',
                                           'kongo',
                                           'tschad',
                                           #'uganda',
                                           #'tiber',
                                           #'giraffe',
                                           'lech',
                                           'rilke',
                                           'dinkel',
                                           #'rosmarin',
                                           #'mosambik',
                                         ],
                               'compute_dir': compute_dir,
                               'worker_file': f'{os.getcwd()}/stn_gpe_healthy_worker.py',
                               'worker_env': "/data/u_rgast_software/anaconda3/envs/pyrates/bin/python3",
                               })

    drop_save_dir = f'{compute_dir}/PopulationDrops/'
    os.makedirs(drop_save_dir, exist_ok=True)

    winner = ga.run(
        initial_gene_pool=pop_genes,
        gene_sampling_func=np.random.normal,
        new_member_sampling_func=np.random.uniform,
        target=[[20, 60, 20],   # healthy control
                [np.nan, 2/3, np.nan],  # ampa blockade in GPe
                [np.nan, 1, np.nan],  # ampa and gabaa blockade in GPe
                [np.nan, 5/3, np.nan],  # GABAA blockade in GPe
                [np.nan, 1/2, np.nan],  # STN blockade
                [2, 5/3, np.nan]  # GABAA blockade in STN
                ],
        max_iter=500,
        enforce_max_iter=True,
        min_fit=1.0,
        n_winners=50,
        n_parent_pairs=150,
        n_new=300,
        sigma_adapt=0.05,
        candidate_save=f'{compute_dir}/GeneticCGSCandidatestn.h5',
        drop_save=drop_save_dir,
        new_pop_on_drop=True,
        pop_save=f'{drop_save_dir}/pop_summary',
        permute=False,
        max_stagnation_steps=10
    )

    #winner.to_hdf(f'{drop_save_dir}/winner.h5', key='data')
