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
        freq_targets = [0.0, np.nan, np.nan, np.nan, np.nan, np.nan, 14.0, 0.0, 14.0, np.nan, np.nan]
        param_grid, invalid_params = eval_params(param_grid)
        conditions = [{},  # healthy control
                      {'k_ie': 0.2},  # AMPA blockade in GPe
                      {'k_ie': 0.2, 'k_ii': 0.2, 'k_str': 0.2},  # AMPA blockade and GABAA blockade in GPe
                      {'k_ii': 0.2, 'k_str': 0.2},  # GABAA blockade in GPe
                      {'k_ie': 0.0},  # STN blockade
                      {'k_ei': 0.2},  # GABAA blocker in STN
                      {},  # PD
                      {'k_ie': 0.2},  # PD + AMPA blockade in GPe
                      {'k_ii': 0.2, 'k_str': 0.2},  # PD + GABAA bloacked in GPe
                      {'k_ee': 0.2},  # PD + AMPA blockade in STN
                      {'k_ei': 0.0}  # PD + GPe blockade
                      ]
        chunk_size = [
            #50,   # carpenters
            150,   # osttimor
            100,   # spanien
            150,   # animals
            50,   # kongo
            #50,   # tschad
            #50,  # uganda
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
                worker_kwargs={'freq_targets': freq_targets, 'targets': target, 'time_lim': 3600.0},
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
        if params.loc[gene_id, 'k_ee'] > 0.2*params.loc[gene_id, 'k_ie']:
            valid = False
        if params.loc[gene_id, 'k_ie'] > 5.0*params.loc[gene_id, 'k_ei']:
            valid = False
        if params.loc[gene_id, 'k_ie'] < 0.2*params.loc[gene_id, 'k_ei']:
            valid = False
        if params.loc[gene_id, 'k_ii'] > 0.8*params.loc[gene_id, 'k_ei']:
            valid = False
        if params.loc[gene_id, 'delta_e'] + params.loc[gene_id, 'delta_e_pd'] < 0:
            valid = False
        if params.loc[gene_id, 'delta_i'] + params.loc[gene_id, 'delta_i_pd'] < 0:
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
        'k_ee': {'min': 0, 'max': 15, 'size': 2, 'sigma': 1.0, 'loc': 5.0, 'scale': 1.0},
        'k_ei': {'min': 0, 'max': 150, 'size': 2, 'sigma': 4.0, 'loc': 75.0, 'scale': 20.0},
        'k_ie': {'min': 0, 'max': 150, 'size': 2, 'sigma': 4.0, 'loc': 75.0, 'scale': 20.0},
        'k_ii': {'min': 0, 'max': 100, 'size': 2, 'sigma': 2.0, 'loc': 50.0, 'scale': 10.0},
        'k_str': {'min': 0, 'max': 400, 'size': 2, 'sigma': 8.0, 'loc': 200.0, 'scale': 50.0},
        'eta_e': {'min': -20, 'max': 20, 'size': 2, 'sigma': 4.0, 'loc': 0.0, 'scale': 5.0},
        'eta_i': {'min': 0, 'max': 40, 'size': 2, 'sigma': 4.0, 'loc': 20.0, 'scale': 5.0},
        'delta_e': {'min': 1.0, 'max': 20.0, 'size': 2, 'sigma': 1.0, 'loc': 8.0, 'scale': 5.0},
        'delta_i': {'min': 1.0, 'max': 20.0, 'size': 2, 'sigma': 1.0, 'loc': 12.0, 'scale': 5.0},
        'tau_e': {'min': 5, 'max': 15, 'size': 2, 'sigma': 0.5, 'loc': 10.0, 'scale': 2.0},
        'tau_i': {'min': 10, 'max': 30, 'size': 2, 'sigma': 0.5, 'loc': 20.0, 'scale': 2.0},
        'tau_ee': {'min': 0.5, 'max': 2.0, 'size': 1, 'sigma': 0.1, 'loc': 1.0, 'scale': 0.2},
        'tau_ee_v': {'min': 0.2, 'max': 2.0, 'size': 1, 'sigma': 0.1, 'loc': 0.5, 'scale': 0.1},
        'tau_ei': {'min': 2.0, 'max': 6.0, 'size': 1, 'sigma': 0.1, 'loc': 4.0, 'scale': 0.5},
        'tau_ei_v': {'min': 0.5, 'max': 2.0, 'size': 1, 'sigma': 0.1, 'loc': 1.0, 'scale': 0.2},
        'tau_ie': {'min': 2.0, 'max': 6.0, 'size': 1, 'sigma': 0.1, 'loc': 4.0, 'scale': 0.5},
        'tau_ie_v': {'min': 0.5, 'max': 2.0, 'size': 1, 'sigma': 0.1, 'loc': 1.0, 'scale': 0.2},
        'tau_ii': {'min': 0.5, 'max': 2.0, 'size': 1, 'sigma': 0.1, 'loc': 1.0, 'scale': 0.2},
        'tau_ii_v': {'min': 0.2, 'max': 2.0, 'size': 1, 'sigma': 0.1, 'loc': 0.5, 'scale': 0.1},
        'k_ee_pd': {'min': 0, 'max': 10, 'size': 1, 'sigma': 0.5, 'loc': 2.0, 'scale': 0.5},
        'k_ei_pd': {'min': 0, 'max': 100, 'size': 1, 'sigma': 1.0, 'loc': 20.0, 'scale': 5.0},
        'k_ie_pd': {'min': 0, 'max': 100, 'size': 1, 'sigma': 1.0, 'loc': 20.0, 'scale': 5.0},
        'k_ii_pd': {'min': 0, 'max': 75, 'size': 1, 'sigma': 0.8, 'loc': 10.0, 'scale': 2.0},
        'k_str_pd': {'min': 0, 'max': 400, 'size': 2, 'sigma': 4.0, 'loc': 200.0, 'scale': 20.0},
        'eta_e_pd': {'min': -20, 'max': 20, 'size': 1, 'sigma': 1.0, 'loc': 0.0, 'scale': 1.0},
        'eta_i_pd': {'min': -30, 'max': 10, 'size': 1, 'sigma': 1.0, 'loc': -10.0, 'scale': 1.0},
        'delta_e_pd': {'min': -10.0, 'max': 0.0, 'size': 1, 'sigma': 0.5, 'loc': -2.0, 'scale': 2.0},
        'delta_i_pd': {'min': -10.0, 'max': 0.0, 'size': 1, 'sigma': 0.5, 'loc': -2.0, 'scale': 2.0},
    }

    param_map = {
        'k_ee': {'vars': ['weight'], 'edges': ['stn/stn']},
        'k_ei': {'vars': ['weight'], 'edges': ['gpe/stn']},
        'k_ie': {'vars': ['weight'], 'edges': ['stn/gpe']},
        'k_ii': {'vars': ['weight'], 'edges': ['gpe/gpe']},
        'k_str': {'vars': ['gpe_proto_op/k_str'], 'nodes': ['gpe']},
        'eta_e': {'vars': ['stn_op/eta_e'], 'nodes': ['stn']},
        'eta_i': {'vars': ['gpe_proto_op/eta_i'], 'nodes': ['gpe']},
        'delta_e': {'vars': ['stn_op/delta_e'], 'nodes': ['stn']},
        'delta_i': {'vars': ['gpe_proto_op/delta_i'], 'nodes': ['gpe']},
        'tau_e': {'vars': ['stn_op/tau_e'], 'nodes': ['stn']},
        'tau_i': {'vars': ['gpe_proto_op/tau_i'], 'nodes': ['gpe']},
        'tau_ee': {'vars': ['delay'], 'edges': ['stn/stn']},
        'tau_ee_v': {'vars': ['spread'], 'edges': ['stn/stn']},
        'tau_ei': {'vars': ['delay'], 'edges': ['gpe/stn']},
        'tau_ei_v': {'vars': ['spread'], 'edges': ['gpe/stn']},
        'tau_ie': {'vars': ['delay'], 'edges': ['stn/gpe']},
        'tau_ie_v': {'vars': ['spread'], 'edges': ['stn/gpe']},
        'tau_ii': {'vars': ['delay'], 'edges': ['gpe/gpe']},
        'tau_ii_v': {'vars': ['spread'], 'edges': ['gpe/gpe']},
        'k_ee_pd': {'vars': ['weight'], 'edges': ['stn/stn']},
        'k_ei_pd': {'vars': ['weight'], 'edges': ['gpe/stn']},
        'k_ie_pd': {'vars': ['weight'], 'edges': ['stn/gpe']},
        'k_ii_pd': {'vars': ['weight'], 'edges': ['gpe/gpe']},
        'k_str_pd': {'vars': ['gpe_proto_op/k_str'], 'nodes': ['gpe']},
        'eta_e_pd': {'vars': ['stn_op/eta_e'], 'nodes': ['stn']},
        'eta_i_pd': {'vars': ['gpe_protp_op/eta_i'], 'nodes': ['gpe']},
        'delta_e_pd': {'vars': ['stn_op/delta_e'], 'nodes': ['stn']},
        'delta_i_pd': {'vars': ['gpe_proto_op/delta_i'], 'nodes': ['gpe']}
    }

    T = 2000.
    dt = 1e-2
    dts = 1e-1
    compute_dir = f"{os.getcwd()}/stn_gpe_combined_opt2"

    # perform genetic optimization
    ga = CustomGOA(fitness_measure=fitness,
                   gs_config={
                       'circuit_template': f"{os.getcwd()}/config/stn_gpe/stn_gpe_basic",
                       'permute_grid': True,
                       'param_map': param_map,
                       'simulation_time': T,
                       'step_size': dt,
                       'sampling_step_size': dts,
                       'inputs': {},
                       'outputs': {'r_e': "stn/stn_op/R_e", 'r_i': 'gpe/gpe_proto_op/R_i'},
                       'init_kwargs': {'backend': 'numpy', 'solver': 'scipy', 'step_size': dt},
                   },
                   cgs_config={'nodes': [
                                         'carpenters',
                                         'osttimor',
                                         'spanien',
                                         'animals',
                                         'kongo',
                                         'tschad',
                                         #'uganda'
                                         ],
                               'compute_dir': compute_dir,
                               'worker_file': f'{os.getcwd()}/stn_gpe_combined_worker.py',
                               'worker_env': "/nobackup/spanien1/rgast/anaconda3/envs/pyrates_test/bin/python3",
                               })

    drop_save_dir = f'{compute_dir}/PopulationDrops/'
    os.makedirs(drop_save_dir, exist_ok=True)

    winner = ga.run(
        initial_gene_pool=pop_genes,
        gene_sampling_func=np.random.normal,
        new_member_sampling_func=np.random.uniform,
        target=[[20, 60],   # healthy control
                [np.nan, 2/3],  # ampa blockade in GPe
                [np.nan, 1],  # ampa and gabaa blockade in GPe
                [np.nan, 5/3],  # GABAA blockade in GPe
                [np.nan, 1/2],  # STN blockade
                [2, 5/3],  # GABAA blockade in STN
                [3/2, 2/3],  # MPTP-induced PD
                [np.nan, 1/3],  # PD + ampa blockade in GPe
                [np.nan, 3/2],  # PD + gabaa blockade in GPe
                [3/2, np.nan],  # PD + AMPA blockade in STN
                [5/2, np.nan],  # PD + GPe blockade
                ],
        max_iter=100,
        enforce_max_iter=True,
        min_fit=1.0,
        n_winners=8,
        n_parent_pairs=200,
        n_new=116,
        sigma_adapt=0.1,
        candidate_save=f'{compute_dir}/GeneticCGSCandidatestn.h5',
        drop_save=drop_save_dir,
        new_pop_on_drop=True,
        pop_save=f'{drop_save_dir}/pop_summary'
    )

    #winner.to_hdf(f'{drop_save_dir}/winner.h5', key='data')
