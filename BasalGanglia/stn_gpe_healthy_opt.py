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
        freq_targets = [0.0, 0.0, 0.0, 0.0, np.nan]
        #param_grid, invalid_params = eval_params(param_grid)
        conditions = [{},  # healthy control
                      {'k_pe': 0.2, 'k_ae': 0.2},  # AMPA blockade in GPe
                      {'k_pe': 0.2, 'k_pp': 0.2, 'k_pa': 0.2, 'k_ae': 0.2, 'k_aa': 0.2, 'k_ap': 0.2,
                       'k_ps': 0.2, 'k_as': 0.2},  # AMPA blockade and GABAA blockade in GPe
                      {'k_pp': 0.2, 'k_pa': 0.2, 'k_aa': 0.2, 'k_ap': 0.2, 'k_ps': 0.2,
                       'k_as': 0.2},  # GABAA blockade in GPe
                      #{'k_pe': 0.0, 'k_ae': 0.0},  # STN blockade
                      #{'k_pe': 0.0, 'k_ae': 0.0, 'k_pp': 0.2, 'k_pa': 0.2, 'k_aa': 0.2, 'k_ap': 0.2,
                      # 'k_ps': 0.2, 'k_as': 0.2},  # STN blockade + GABAA blockade in GPe
                      {'k_ep': 0.2}  # GABAA blocker in STN
                      ]
        param_scalings = [
            ('delta_e', 'tau_e', 2.0),
            ('delta_p', 'tau_p', 2.0),
            ('delta_a', 'tau_a', 2.0),
            ('k_ee', 'delta_e', 0.5),
            ('k_ep', 'delta_e', 0.5),
            ('k_pe', 'delta_p', 0.5),
            ('k_pp', 'delta_p', 0.5),
            ('k_pa', 'delta_p', 0.5),
            ('k_ps', 'delta_p', 0.5),
            ('k_ae', 'delta_a', 0.5),
            ('k_ap', 'delta_a', 0.5),
            ('k_aa', 'delta_a', 0.5),
            ('k_as', 'delta_a', 0.5),
            ('eta_e', 'delta_e', 1.0),
            ('eta_p', 'delta_p', 1.0),
            ('eta_a', 'delta_a', 1.0),
            ]
        chunk_size = [
            #60,  # carpenters
            60,  # osttimor
            40,  # spanien
            #100,  # animals
            40,  # kongo
            40,  # tschad
            60,  # uganda
            60,  # tiber
            60,  # giraffe
            40,  # lech
            20,  # rilke
            20,  # dinkel
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
                worker_kwargs={'freq_targets': freq_targets, 'targets': target, 'time_lim': 5400.0, 'cpu_lim': False,
                               'nproc_lim': False, 'memory_lim': False},
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
    diff = np.asarray([0.0 if np.isnan(t_tmp) else y_tmp - t_tmp for y_tmp, t_tmp in zip(y, t)]).flatten()**2
    t[np.isnan(t)] = 1.0
    t[t == 0] = 1.0
    weights = 1/np.abs(t)
    return weights @ diff


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    pop_size = 2048
    pop_genes = {
        'k_ee': {'min': 0, 'max': 8, 'size': pop_size, 'sigma': 0.1, 'loc': 3.0, 'scale': 1.0},
        'k_ae': {'min': 10, 'max': 100, 'size': pop_size, 'sigma': 0.5, 'loc': 30.0, 'scale': 4.0},
        'k_pe': {'min': 20, 'max': 200, 'size': pop_size, 'sigma': 1.0, 'loc': 80.0, 'scale': 8.0},
        'k_pp': {'min': 2, 'max': 12, 'size': pop_size, 'sigma': 0.2, 'loc': 6.0, 'scale': 1.0},
        'k_ep': {'min': 10, 'max': 100, 'size': pop_size, 'sigma': 0.5, 'loc': 30.0, 'scale': 3.0},
        'k_ap': {'min': 6, 'max': 60, 'size': pop_size, 'sigma': 0.5, 'loc': 30.0, 'scale': 3.0},
        'k_aa': {'min': 0, 'max': 8, 'size': pop_size, 'sigma': 0.1, 'loc': 4.0, 'scale': 1.0},
        'k_pa': {'min': 20, 'max': 200, 'size': pop_size, 'sigma': 0.8, 'loc': 60.0, 'scale': 6.0},
        'k_ps': {'min': 20, 'max': 200, 'size': pop_size, 'sigma': 1.0, 'loc': 80.0, 'scale': 8.0},
        'k_as': {'min': 40, 'max': 300, 'size': pop_size, 'sigma': 2.0, 'loc': 160.0, 'scale': 16.0},
        'eta_e': {'min': -4, 'max': 2, 'size': pop_size, 'sigma': 0.1, 'loc': 0.0, 'scale': 1.0},
        'eta_p': {'min': -4, 'max': 2, 'size': pop_size, 'sigma': 0.1, 'loc': 0.0, 'scale': 1.0},
        'eta_a': {'min': -4, 'max': 2, 'size': pop_size, 'sigma': 0.1, 'loc': 0.0, 'scale': 1.0},
        'delta_e': {'min': 0.05, 'max': 0.15, 'size': pop_size, 'sigma': 0.01, 'loc': 0.1, 'scale': 0.05},
        'delta_p': {'min': 0.1, 'max': 0.5, 'size': pop_size, 'sigma': 0.02, 'loc': 0.3, 'scale': 0.05},
        'delta_a': {'min': 0.1, 'max': 0.4, 'size': pop_size, 'sigma': 0.02, 'loc': 0.2, 'scale': 0.05},
        'tau_e': {'min': 13, 'max': 13, 'size': pop_size, 'sigma': 0.0, 'loc': 13.0, 'scale': 0.0},
        'tau_p': {'min': 25, 'max': 25, 'size': pop_size, 'sigma': 0.0, 'loc': 25.0, 'scale': 0.0},
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
        'k_ps': {'vars': ['weight'], 'edges': [('str', 'gpe_p')]},
        'k_as': {'vars': ['weight'], 'edges': [('str', 'gpe_a')]},
        'eta_e': {'vars': ['stn_op/eta_e'], 'nodes': ['stn']},
        'eta_p': {'vars': ['gpe_proto_op/eta_i'], 'nodes': ['gpe_p']},
        'eta_a': {'vars': ['gpe_arky_op/eta_a'], 'nodes': ['gpe_a']},
        'delta_e': {'vars': ['stn_op/delta_e'], 'nodes': ['stn']},
        'delta_p': {'vars': ['gpe_proto_op/delta_i'], 'nodes': ['gpe_p']},
        'delta_a': {'vars': ['gpe_arky_op/delta_a'], 'nodes': ['gpe_a']},
        'tau_e': {'vars': ['stn_op/tau_e'], 'nodes': ['stn']},
        'tau_p': {'vars': ['gpe_proto_op/tau_i'], 'nodes': ['gpe_p']},
        'tau_a': {'vars': ['gpe_arky_op/tau_a'], 'nodes': ['gpe_a']},
    }

    T = 500.
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
                                           #'carpenters',
                                           'osttimor',
                                           'spanien',
                                           #'animals',
                                           'kongo',
                                           'tschad',
                                           'uganda',
                                           'tiber',
                                           'giraffe',
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
        target=[[19, 62, 35],  # healthy control
                [np.nan, 35, np.nan],  # ampa blockade in GPe
                [np.nan, 76, np.nan],  # ampa and gabaa blockade in GPe
                [np.nan, 135, np.nan],  # GABAA blockade in GPe
                #[np.nan, 30, np.nan],  # STN blockade
                #[np.nan, 60, np.nan],  # STN blockade + gabaa blockade in GPe
                [40, 124, np.nan]  # GABAA blockade in STN
                ],
        max_iter=500,
        enforce_max_iter=True,
        min_fit=1.5,
        n_winners=30,
        n_parent_pairs=120,
        n_new=248,
        sigma_adapt=0.05,
        candidate_save=f'{compute_dir}/GeneticCGSCandidatestn.h5',
        drop_save=drop_save_dir,
        new_pop_on_drop=True,
        pop_save=f'{drop_save_dir}/pop_summary',
        permute=False,
        max_stagnation_steps=5,
        stagnation_decimals=3
    )
    #winner.to_hdf(f'{drop_save_dir}/winner.h5', key='data')
