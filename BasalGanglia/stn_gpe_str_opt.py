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
        result_vars = ['r_e', 'r_p', 'r_a', 'r_s']
        freq_targets = [0.0, np.nan, np.nan, np.nan, np.nan]
        param_grid, invalid_params = eval_params(param_grid)
        conditions = [{},  # healthy control
                      {'k_pe': 0.2, 'k_ae': 0.2},  # AMPA blockade in GPe
                      {'k_pe': 0.2, 'k_ae': 0.2, 'k_pp': 0.2, 'k_pa': 0.2, 'k_ps': 0.2, 'k_aa': 0.2, 'k_ap': 0.2,
                       'k_as': 0.2},  # AMPA blockade and GABAA blockade in GPe
                      {'k_pp': 0.2, 'k_pa': 0.2, 'k_ps': 0.2, 'k_aa': 0.2, 'k_ap': 0.2,
                       'k_as': 0.2},  # GABAA blockade in GPe
                      {'k_pe': 0.0, 'k_ae': 0.0},  # STN blockade
                      {'k_ep': 0.2},  # GABAA blocker in STN
                      ]
        chunk_size = [
            50,  # carpenters
            100,  # osttimor
            50,  # spanien
            100,  # animals
            50,  # kongo
            50,  # tschad
            100,  # uganda
            # 50,  # tiber
            50,  # giraffe
            20,  # lech
            10,  # rilke
            10,  # dinkel
            10,  # rosmarin
            10,  # mosambik
            # 50,  # compute servers
            # 40,
            # 30,
            # 20,
            # 10,
            # 50,
            # 40,
            # 30,
            # 20,
            # 10,
            # 50,
            # 40,
            # 30,
            # 20,
            # 10,
            # 50,
            # 40,
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
                worker_kwargs={'targets': target, 'time_lim': 2000.0, 'freq_targets': freq_targets},
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
    weights = 1 / np.abs(t)
    return weights @ np.abs(diff)


def eval_params(params):
    valid_params = []
    invalid_params = []
    for i, gene_id in enumerate(params.index):

        # check validity conditions
        valid = True
        if params.loc[gene_id, 'k_ee'] > 0.2 * params.loc[gene_id, 'k_pe']:
            valid = False
        if params.loc[gene_id, 'k_pe'] > 5.0 * params.loc[gene_id, 'k_ep']:
            valid = False
        if params.loc[gene_id, 'k_ep'] > 5.0 * params.loc[gene_id, 'k_pe']:
            valid = False
        if params.loc[gene_id, 'k_pp'] > 0.8 * params.loc[gene_id, 'k_ep']:
            valid = False
        if params.loc[gene_id, 'k_ps'] > 0.5 * params.loc[gene_id, 'k_as']:
            valid = False
        if params.loc[gene_id, 'delta_p'] > params.loc[gene_id, 'delta_a']:
            valid = False
        #if params.loc[gene_id, 'tau_a'] > params.loc[gene_id, 'tau_p']:
        #    valid = False
        #if params.loc[gene_id, 'tau_e'] > params.loc[gene_id, 'tau_a']:
        #    valid = False

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

    pop_size = 2048
    pop_genes = {
        'k_ee': {'min': 0, 'max': 15, 'size': pop_size, 'sigma': 0.5, 'loc': 5.0, 'scale': 1.0},
        'k_ae': {'min': 0, 'max': 150, 'size': pop_size, 'sigma': 2.0, 'loc': 75.0, 'scale': 10.0},
        'k_pe': {'min': 0, 'max': 150, 'size': pop_size, 'sigma': 2.0, 'loc': 75.0, 'scale': 10.0},
        'k_pp': {'min': 0, 'max': 100, 'size': pop_size, 'sigma': 2.0, 'loc': 20.0, 'scale': 2.0},
        'k_ep': {'min': 0, 'max': 150, 'size': pop_size, 'sigma': 2.0, 'loc': 50.0, 'scale': 5.0},
        'k_ap': {'min': 0, 'max': 100, 'size': pop_size, 'sigma': 2.0, 'loc': 20.0, 'scale': 2.0},
        'k_aa': {'min': 0, 'max': 100, 'size': pop_size, 'sigma': 2.0, 'loc': 20.0, 'scale': 2.0},
        'k_pa': {'min': 0, 'max': 100, 'size': pop_size, 'sigma': 2.0, 'loc': 20.0, 'scale': 2.0},
        'k_sa': {'min': 0, 'max': 150, 'size': pop_size, 'sigma': 2.0, 'loc': 50.0, 'scale': 5.0},
        'k_ss': {'min': 0, 'max': 200, 'size': pop_size, 'sigma': 2.0, 'loc': 100.0, 'scale': 10.0},
        'k_as': {'min': 0, 'max': 400, 'size': pop_size, 'sigma': 2.0, 'loc': 200.0, 'scale': 10.0},
        'k_ps': {'min': 0, 'max': 200, 'size': pop_size, 'sigma': 2.0, 'loc': 100.0, 'scale': 10.0},
        'eta_e': {'min': -30, 'max': 30, 'size': pop_size, 'sigma': 1.0, 'loc': -5.0, 'scale': 5.0},
        'eta_p': {'min': 0, 'max': 60, 'size': pop_size, 'sigma': 1.0, 'loc': 20.0, 'scale': 5.0},
        'eta_a': {'min': -20, 'max': 40, 'size': pop_size, 'sigma': 1.0, 'loc': 10.0, 'scale': 5.0},
        'eta_s': {'min': -100, 'max': 0, 'size': pop_size, 'sigma': 1.0, 'loc': -40.0, 'scale': 5.0},
        'delta_e': {'min': 0.5, 'max': 20.0, 'size': pop_size, 'sigma': 0.5, 'loc': 4.0, 'scale': 1.0},
        'delta_p': {'min': 1.0, 'max': 20.0, 'size': pop_size, 'sigma': 0.5, 'loc': 6.0, 'scale': 1.0},
        'delta_a': {'min': 1.0, 'max': 20.0, 'size': pop_size, 'sigma': 0.5, 'loc': 8.0, 'scale': 1.0},
        'delta_s': {'min': 0.5, 'max': 20.0, 'size': pop_size, 'sigma': 0.2, 'loc': 2.0, 'scale': 0.5},
        #'tau_e': {'min': 5, 'max': 15, 'size': pop_size, 'sigma': 0.5, 'loc': 10.0, 'scale': 2.0},
        #'tau_p': {'min': 15, 'max': 30, 'size': pop_size, 'sigma': 0.5, 'loc': 24.0, 'scale': 2.0},
        #'tau_a': {'min': 10, 'max': 25, 'size': pop_size, 'sigma': 0.5, 'loc': 20.0, 'scale': 2.0},
        #'tau_s': {'min': 10, 'max': 30, 'size': pop_size, 'sigma': 0.5, 'loc': 20.0, 'scale': 2.0},
        #'tau_ee_v': {'min': 0.5, 'max': 1.0, 'size': 2, 'sigma': 0.1, 'loc': 0.5, 'scale': 0.1},
        # 'tau_ei': {'min': 3.0, 'max': 5.0, 'size': 1, 'sigma': 0.1, 'loc': 4.0, 'scale': 0.1},
        #'tau_ei_v': {'min': 0.5, 'max': 1.0, 'size': 2, 'sigma': 0.1, 'loc': 1.0, 'scale': 0.2},
        # 'tau_ie': {'min': 2.0, 'max': 4.0, 'size': 1, 'sigma': 0.1, 'loc': 3.0, 'scale': 0.1},
        #'tau_ie_v': {'min': 0.8, 'max': 1.6, 'size': 2, 'sigma': 0.1, 'loc': 0.7, 'scale': 0.1},
        #'tau_ii_v': {'min': 0.5, 'max': 1.0, 'size': 2, 'sigma': 0.1, 'loc': 0.5, 'scale': 0.1},
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
        'k_sa': {'vars': ['weight'], 'edges': [('gpe_a', 'str')]},
        'k_ss': {'vars': ['weight'], 'edges': [('str', 'str')]},
        'k_as': {'vars': ['weight'], 'edges': [('str', 'gpe_a')]},
        'k_ps': {'vars': ['weight'], 'edges': [('str', 'gpe_p')]},
        'eta_e': {'vars': ['stn_op/eta_e'], 'nodes': ['stn']},
        'eta_p': {'vars': ['gpe_proto_op/eta_i'], 'nodes': ['gpe_p']},
        'eta_a': {'vars': ['gpe_arky_op/eta_a'], 'nodes': ['gpe_a']},
        'eta_s': {'vars': ['str_op/eta_s'], 'nodes': ['str']},
        'delta_e': {'vars': ['stn_op/delta_e'], 'nodes': ['stn']},
        'delta_p': {'vars': ['gpe_proto_op/delta_i'], 'nodes': ['gpe_p']},
        'delta_a': {'vars': ['gpe_arky_op/delta_a'], 'nodes': ['gpe_a']},
        'delta_s': {'vars': ['str_op/delta_s'], 'nodes': ['str']},
        'tau_e': {'vars': ['stn_op/tau_e'], 'nodes': ['stn']},
        'tau_p': {'vars': ['gpe_proto_op/tau_i'], 'nodes': ['gpe_p']},
        'tau_a': {'vars': ['gpe_arky_op/tau_a'], 'nodes': ['gpe_a']},
        'tau_s': {'vars': ['str_op/tau_s'], 'nodes': ['str']},
    }

    T = 2000.
    dt = 1e-2
    dts = 1e-1
    compute_dir = f"{os.getcwd()}/stn_gpe_str_opt"

    # perform genetic optimization
    ga = CustomGOA(fitness_measure=fitness,
                   gs_config={
                       'circuit_template': f"{os.getcwd()}/config/stn_gpe/stn_gpe_str",
                       'permute_grid': True,
                       'param_map': param_map,
                       'simulation_time': T,
                       'step_size': dt,
                       'sampling_step_size': dts,
                       'inputs': {},
                       'outputs': {'r_e': "stn/stn_op/R_e", 'r_p': 'gpe_p/gpe_proto_op/R_i',
                                   'r_a': 'gpe_a/gpe_arky_op/R_a', 'r_s': 'str/str_op/R_s'},
                       'init_kwargs': {'backend': 'numpy', 'solver': 'scipy', 'step_size': dt},
                   },
                   cgs_config={'nodes': [
                       'carpenters',
                       'osttimor',
                       'spanien',
                       'animals',
                       'kongo',
                       'tschad',
                       'uganda',
                       # 'tiber',
                       'giraffe',
                       'lech',
                       'rilke',
                       'dinkel',
                       'rosmarin',
                       'mosambik',
                       # 'comps06h01',
                       # 'comps06h02',
                       # 'comps06h03',
                       # 'comps06h04',
                       # 'comps06h05',
                       # 'comps06h06',
                       # 'comps06h07',
                       # 'comps06h08',
                       # 'comps06h09',
                       # 'comps06h10',
                       # 'comps06h11',
                       # 'comps06h12',
                       # 'comps06h13',
                       # 'comps06h14',
                       # 'scorpions',
                       # 'spliff',
                       # 'supertramp',
                       # 'ufo'
                   ],
                       'compute_dir': compute_dir,
                       'worker_file': f'{os.getcwd()}/stn_gpe_str_worker.py',
                       'worker_env': "/data/u_rgast_software/anaconda3/envs/pyrates/bin/python3",
                   })

    drop_save_dir = f'{compute_dir}/PopulationDrops/'
    os.makedirs(drop_save_dir, exist_ok=True)

    winner = ga.run(
        initial_gene_pool=pop_genes,
        gene_sampling_func=np.random.normal,
        new_member_sampling_func=np.random.normal,
        target=[[20, 60, 20, 2],  # healthy control
                [np.nan, 2/3, np.nan, np.nan],  # ampa blockade in GPe
                [np.nan, 1, np.nan, np.nan],  # ampa and gabaa blockade in GPe
                [np.nan, 5/3, np.nan, np.nan],  # GABAA blockade in GPe
                [np.nan, 1/2, np.nan, np.nan],  # STN blockade
                [5/3, 5/3, np.nan, np.nan],  # GABAA blockade in STN
                ],
        max_iter=100,
        enforce_max_iter=True,
        min_fit=1.0,
        n_winners=10,
        n_parent_pairs=300,
        n_new=200,
        sigma_adapt=0.05,
        candidate_save=f'{compute_dir}/GeneticCGSCandidatestn.h5',
        drop_save=drop_save_dir,
        new_pop_on_drop=True,
        pop_save=f'{drop_save_dir}/pop_summary',
        permute=False
    )

    # winner.to_hdf(f'{drop_save_dir}/winner.h5', key='data')
