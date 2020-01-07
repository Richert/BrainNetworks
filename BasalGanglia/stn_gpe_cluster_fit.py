import os
import warnings
import numpy as np
from pyrates.utility.genetic_algorithm import CGSGeneticAlgorithm
from pyrates.utility import welch
from pandas import DataFrame, read_hdf


class CustomGOA(CGSGeneticAlgorithm):

    def eval_fitness(self, target: list, **kwargs):

        # define simulation conditions
        worker_file = self.cgs_config['worker_file'] if 'worker_file' in self.cgs_config else None
        param_grid = self.pop.drop(['fitness', 'sigma', 'results'], axis=1)
        models_vars = ['k_ie', 'k_ii', 'k_ei', 'k_ee', 'eta_e', 'eta_i', 'eta_str', 'eta_tha', 'alpha',
                       'delta_e', 'delta_i']
        freq_targets = [0.0, 0.0, 0.0, 0.0, 50.0, 0.0, 13.0]
        param_grid, invalid_params = eval_params(param_grid)
        zero_vec = [0.0 for _ in range(param_grid.shape[0])]
        conditions = [{},  # healthy control
                      {'k_ie': zero_vec},  # STN blockade
                      {'k_ii': zero_vec, 'eta_str': zero_vec},  # GABAA blockade in GPe
                      {'k_ie': zero_vec, 'k_ii': zero_vec, 'eta_str': zero_vec},
                      # STN blockade and GABAA blockade in GPe
                      {'k_ie': zero_vec, 'eta_tha': zero_vec},  # AMPA + NMDA blocker in GPe
                      {'k_ei': zero_vec},  # GABAA antagonist in STN
                      {'k_ei': param_grid['k_ei'] + param_grid['k_ei_pd'],
                       'k_ie': param_grid['k_ie'] + param_grid['k_ie_pd'],
                       'k_ee': param_grid['k_ee'] + param_grid['k_ee_pd'],
                       'k_ii': param_grid['k_ii'] + param_grid['k_ii_pd'],
                       'eta_e': param_grid['eta_e'] + param_grid['eta_e_pd'],
                       'eta_i': param_grid['eta_i'] + param_grid['eta_i_pd'],
                       'eta_str': param_grid['eta_str'] + param_grid['eta_str_pd'],
                       'eta_tha': param_grid['eta_tha'] + param_grid['eta_tha_pd'],
                       'delta_e': param_grid['delta_e'] + param_grid['delta_e_pd'],
                       'delta_i': param_grid['delta_i'] + param_grid['delta_i_pd'],
                       }  # parkinsonian condition
                      ]
        chunk_size = [
            260,   # animals
            220,  # spanien
            60,   # kongo
            60,   # tschad
            120,   # ostimor
            260,   # carpenters
            60,   # uganda
        ]

        # perform simulations
        if len(param_grid) > 0:
            self.gs_config['init_kwargs'].update(kwargs)
            res_file = self.cgs.run(
                circuit_template=self.gs_config['circuit_template'],
                param_grid=param_grid,
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
                           'model_vars': models_vars},
                worker_kwargs={'param_grid': param_grid, 'freq_targets': freq_targets, 'targets': target})
            try:
                fitness_tmp = read_hdf(res_file, key=f'Results/fitness')
            except FileNotFoundError:
                fitness_tmp = np.inf

            # calculate fitness
            for gene_id in param_grid.index:
                self.pop.at[gene_id, 'fitness'] = 1.0 / fitness_tmp

        # set fitness of invalid parametrizations
        for gene_id in invalid_params.index:
            self.pop.at[gene_id, 'fitness'] = 0.0


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
        if params.loc[gene_id, 'k_ee_pd'] < 0.0:
            valid = False
        if params.loc[gene_id, 'k_ei_pd'] < 0.0:
            valid = False
        if params.loc[gene_id, 'k_ie_pd'] < 0.0:
            valid = False
        if params.loc[gene_id, 'k_ii_pd'] < 0.0:
            valid = False
        if params.loc[gene_id, 'eta_str_pd'] > 0.0:
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
        'k_ee': {'min': 0, 'max': 100, 'N': 2, 'sigma': 0.4},
        'k_ei': {'min': 0, 'max': 300, 'N': 2, 'sigma': 0.8},
        'k_ie': {'min': 0, 'max': 300, 'N': 2, 'sigma': 0.8},
        'k_ii': {'min': 0, 'max': 200, 'N': 2, 'sigma': 0.8},
        'eta_e': {'min': -30, 'max': 30, 'N': 2, 'sigma': 0.4},
        'eta_i': {'min': -30, 'max': 30, 'N': 2, 'sigma': 0.4},
        'eta_str': {'min': -30, 'max': 0, 'N': 2, 'sigma': 0.4},
        'eta_tha': {'min': 0, 'max': 30, 'N': 2, 'sigma': 0.4},
        'alpha': {'min': 0, 'max': 20.0, 'N': 2, 'sigma': 0.2},
        'delta_e': {'min': 0.1, 'max': 5.0, 'N': 2, 'sigma': 0.2},
        'delta_i': {'min': 0.1, 'max': 5.0, 'N': 2, 'sigma': 0.2},
        'k_ee_pd': {'min': 0, 'max': 50, 'N': 1, 'sigma': 0.4},
        'k_ei_pd': {'min': 0, 'max': 200, 'N': 1, 'sigma': 0.8},
        'k_ie_pd': {'min': 0, 'max': 200, 'N': 1, 'sigma': 0.8},
        'k_ii_pd': {'min': 0, 'max': 100, 'N': 1, 'sigma': 0.8},
        'eta_e_pd': {'min': -10, 'max': 10, 'N': 1, 'sigma': 0.4},
        'eta_i_pd': {'min': -10, 'max': 10, 'N': 1, 'sigma': 0.4},
        'eta_str_pd': {'min': -20, 'max': 0, 'N': 1, 'sigma': 0.4},
        'eta_tha_pd': {'min': -10.0, 'max': 10, 'N': 1, 'sigma': 0.4},
        'delta_e_pd': {'min': -4.0, 'max': 0.0, 'N': 1, 'sigma': 0.2},
        'delta_i_pd': {'min': -4.0, 'max': 0.0, 'N': 1, 'sigma': 0.2},
    }

    param_map = {
        'k_ee': {'vars': ['qif_full/k_ee'], 'nodes': ['stn_gpe']},
        'k_ei': {'vars': ['qif_full/k_ei'], 'nodes': ['stn_gpe']},
        'k_ie': {'vars': ['qif_full/k_ie'], 'nodes': ['stn_gpe']},
        'k_ii': {'vars': ['qif_full/k_ii'], 'nodes': ['stn_gpe']},
        'eta_e': {'vars': ['qif_full/eta_e'], 'nodes': ['stn_gpe']},
        'eta_i': {'vars': ['qif_full/eta_i'], 'nodes': ['stn_gpe']},
        'eta_str': {'vars': ['qif_full/eta_str'], 'nodes': ['stn_gpe']},
        'eta_tha': {'vars': ['qif_full/eta_tha'], 'nodes': ['stn_gpe']},
        'alpha': {'vars': ['qif_full/alpha'], 'nodes': ['stn_gpe']},
        'delta_e': {'vars': ['qif_full/delta_e'], 'nodes': ['stn_gpe']},
        'delta_i': {'vars': ['qif_full/delta_i'], 'nodes': ['stn_gpe']}
    }

    T = 10000.
    dt = 1e-2
    dts = 1e-1
    compute_dir = f"{os.getcwd()}/results"

    ga = CustomGOA(fitness_measure=fitness,
                   gs_config={
                       'circuit_template': f"{os.getcwd()}/config/stn_gpe/net_stn_gpe",
                       'permute_grid': True,
                       'param_map': param_map,
                       'simulation_time': T,
                       'step_size': dt,
                       'sampling_step_size': dts,
                       'inputs': {},
                       'outputs': {'r_e': "stn_gpe/qif_full/R_e", 'r_i': 'stn_gpe/qif_full/R_i'},
                       'init_kwargs': {'backend': 'numpy', 'solver': 'scipy', 'step_size': dt},
                   },
                   cgs_config={'nodes': ['animals', 'spanien', 'kongo', 'tschad',
                                         'osttimor', 'carpenters', 'uganda'
                                         ],
                               'compute_dir': compute_dir,
                               'worker_file': f'{os.getcwd()}/stn_gpe_worker.py',
                               'worker_env': "/nobackup/spanien1/rgast/anaconda3/envs/pyrates_test/bin/python3",
                               })

    drop_save_dir = f'{compute_dir}/PopulationDrops/'
    os.makedirs(drop_save_dir, exist_ok=True)

    winner = ga.run(
        initial_gene_pool=pop_genes,
        gene_sampling_func=np.random.uniform,
        target=[[20, 60],   # healthy control
                [np.nan, 40],  # stn blockade
                [np.nan, 90],  # gabaa blockade in GPe
                [np.nan, 60],  # STN blockade and GABAA blockade in GPe
                [np.nan, 17],  # blockade of all excitatory inputs to GPe
                [35, 80],  # GABAA antagonist in STN
                [30, 40]   # parkinsonian condition
                ],
        max_iter=200,
        min_fit=0.1,
        n_winners=20,
        n_parent_pairs=1800,
        n_new=228,
        sigma_adapt=0.015,
        candidate_save=f'{compute_dir}/GeneticCGSCandidate.h5',
        drop_save=drop_save_dir,
        new_pop_on_drop=True,
    )

    winner.to_hdf(f'{compute_dir}/PopulationDrops/winner.h5', key='data')