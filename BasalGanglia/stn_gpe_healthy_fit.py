import os
import warnings
import pandas as pd
import numpy as np
from pyrates.utility.genetic_algorithm import GSGeneticAlgorithm
from pyrates.utility import grid_search, welch


class CustomGOA(GSGeneticAlgorithm):

    def eval_fitness(self, target: list, **kwargs):

        # define simulation conditions
        param_grid = self.pop.drop(['fitness', 'sigma'], axis=1)
        results = []
        conditions = [{},  # healthy control
                      {'k_ie': [0.0]},  # STN blockade
                      {'k_ii': [0.0], 'eta_str': [0.0]},  # GABAA blockade in GPe
                      {'k_ie': [0.0], 'k_ii': [0.0], 'eta_str': [0.0]},  # STN blockade and GABAA blockade in GPe
                      {'k_ie': [0.0], 'eta_tha': [0.0]},  # AMPA + NMDA blocker in GPe
                      {'k_ei': [0.0]},  # GABAA antagonist in STN
                      {'k_ei': param_grid['k_ei_pd'],
                       'k_ie': param_grid['k_ie_pd'],
                       'k_ee': param_grid['k_ee_pd'],
                       'k_ii': param_grid['k_ii_pd'],
                       'eta_e': param_grid['eta_e_pd'],
                       'eta_i': param_grid['eta_i_pd'],
                       'eta_str': param_grid['eta_str_pd'],
                       'eta_tha': param_grid['eta_tha_pd'],
                       }   # parkinsonian condition
                      ]
        freq_targets = [0.0, 0.0, 0.0, 70.0, 0.0, 13.0]
        param_grid, invalid_params = eval_params(param_grid)

        # perform simulations
        for c_dict in conditions:

            param_grid_tmp = param_grid.copy()
            param_grid_tmp.update(c_dict)
            results_tmp, params_tmp = grid_search(circuit_template=self.gs_config['circuit_template'],
                                                  param_grid=param_grid_tmp,
                                                  param_map=self.gs_config['param_map'],
                                                  simulation_time=self.gs_config['simulation_time'],
                                                  dt=self.gs_config['step_size'],
                                                  sampling_step_size=self.gs_config['sampling_step_size'],
                                                  permute_grid=False,
                                                  inputs=self.gs_config['inputs'],
                                                  outputs=self.gs_config['outputs'].copy(),
                                                  init_kwargs=self.gs_config['init_kwargs'],
                                                  **kwargs
                                                  )
            results.append(results_tmp)

        # calculate fitness
        for i, gene_id in enumerate(param_grid.index):
            outputs, freq, pow = [], [], []
            for j, r in enumerate(results):
                idx = np.argwhere(r.columns.get_level_values(1).values == gene_id).squeeze()
                outputs.append([np.mean(r.loc[4.0:, :][idx]['r_e']),
                                np.mean(r.loc[4.0:, :][idx]['r_i'])])
                psds, freqs = welch(r.iloc[:, idx]['r_i'], tmin=0.1, fmin=1.0, fmax=100.0)
                freq.append(freqs)
                pow.append(psds)
            dist1 = self.fitness_measure(outputs, target, **self.fitness_kwargs)
            dist2 = analyze_oscillations(freq_targets, freq, pow)
            self.pop.at[i, 'fitness'] = float(1 / (dist1 + dist2))
            self.pop.at[i, 'outputs'] = outputs

        # set fitness of invalid parametrizations
        for i, gene_id in enumerate(invalid_params.index):
            self.pop.at[i, 'fitness'] = -1e8


def fitness(y, t):
    y = np.asarray(y).flatten()
    t = np.asarray(t).flatten()
    diff = np.asarray([0.0 if t_tmp is None else y_tmp - t_tmp for y_tmp, t_tmp in zip(y, t)])
    return np.sqrt(np.mean(diff**2))


def analyze_oscillations(freq_targets, freqs, pows):
    dist = []
    for t, f, p in zip(freq_targets, freqs, pows):
        if t:
            f_tmp = f[np.argmax(pows)]
            dist.append(f_tmp - f)
        else:
            p_tmp = np.max(pows)
            dist.append(-p_tmp)
    return fitness(dist, freq_targets)


def eval_params(params):
    valid_params = []
    invalid_params = []
    for i, gene_id in enumerate(params.index):
        valid = True
        if params.loc[gene_id, 'k_ee'] > 0.25*params.loc[gene_id, 'k_ie']:
            valid = False
        if params.loc[gene_id, 'k_ii'] > params.loc[gene_id, 'k_ei']:
            valid = False
        if params.loc[gene_id, 'k_ie'] > 0.75*params.loc[gene_id, 'k_ei']:
            valid = False
        if valid:
            valid_params.append(gene_id)
        else:
            invalid_params.append(gene_id)

    return params.loc[valid_params, :], params.loc[invalid_params, :]


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    pop_genes = {
        'k_ee': {'min': 0, 'max': 20, 'N': 2, 'sigma': 0.4},
        'k_ei': {'min': 10, 'max': 100, 'N': 2, 'sigma': 0.8},
        'k_ie': {'min': 10, 'max': 100, 'N': 2, 'sigma': 0.8},
        'k_ii': {'min': 0, 'max': 50, 'N': 2, 'sigma': 0.8},
        'eta_e': {'min': -10, 'max': 10, 'N': 2, 'sigma': 0.4},
        'eta_i': {'min': -10, 'max': 10, 'N': 2, 'sigma': 0.4},
        'eta_str': {'min': -10, 'max': 0, 'N': 2, 'sigma': 0.4},
        'eta_tha': {'min': 0, 'max': 10, 'N': 2, 'sigma': 0.4},
        'alpha': {'min': 0, 'max': 10.0, 'N': 1, 'sigma': 0.2},
        'k_ee_pd': {'min': 0, 'max': 20, 'N': 1, 'sigma': 0.4},
        'k_ei_pd': {'min': 10, 'max': 100, 'N': 1, 'sigma': 0.8},
        'k_ie_pd': {'min': 10, 'max': 100, 'N': 1, 'sigma': 0.8},
        'k_ii_pd': {'min': 0, 'max': 50, 'N': 1, 'sigma': 0.8},
        'eta_e_pd': {'min': -10, 'max': 10, 'N': 1, 'sigma': 0.4},
        'eta_i_pd': {'min': -10, 'max': 10, 'N': 1, 'sigma': 0.4},
        'eta_str_pd': {'min': -10, 'max': 0, 'N': 1, 'sigma': 0.4},
        'eta_tha_pd': {'min': 0, 'max': 10, 'N': 1, 'sigma': 0.4},
    }

    param_map = {
        'k_ee': {'vars': ['qif_stn/k_ee'], 'nodes': ['stn']},
        'k_ei': {'vars': ['qif_stn/k_ei'], 'nodes': ['stn']},
        'k_ie': {'vars': ['qif_gpe/k_ie'], 'nodes': ['gpe']},
        'k_ii': {'vars': ['qif_gpe/k_ii'], 'nodes': ['gpe']},
        'eta_e': {'vars': ['qif_stn/eta_e'], 'nodes': ['stn']},
        'eta_i': {'vars': ['qif_gpe/eta_i'], 'nodes': ['gpe']},
        'eta_str': {'vars': ['qif_gpe/eta_i'], 'nodes': ['gpe']},
        'eta_tha': {'vars': ['qif_gpe/eta_i'], 'nodes': ['gpe']},
        'alpha': {'vars': ['qif_gpe/alpha'], 'nodes': ['gpe']},
        'k_ee_pd': {'vars': ['qif_stn/k_ee'], 'nodes': ['stn']},
        'k_ei_pd': {'vars': ['qif_stn/k_ei'], 'nodes': ['stn']},
        'k_ie_pd': {'vars': ['qif_gpe/k_ie'], 'nodes': ['gpe']},
        'k_ii_pd': {'vars': ['qif_gpe/k_ii'], 'nodes': ['gpe']},
        'eta_e_pd': {'vars': ['qif_stn/eta_e'], 'nodes': ['stn']},
        'eta_i_pd': {'vars': ['qif_gpe/eta_i'], 'nodes': ['gpe']},
        'eta_str_pd': {'vars': ['qif_gpe/eta_i'], 'nodes': ['gpe']},
        'eta_tha_pd': {'vars': ['qif_gpe/eta_i'], 'nodes': ['gpe']},
        'alpha_pd': {'vars': ['qif_gpe/alpha'], 'nodes': ['gpe']}
    }

    T = 5.
    dt = 1e-4
    dts = 1e-3

    compute_dir = "results"

    ga = CustomGOA(fitness_measure=fitness,
                   gs_config={
                       'circuit_template': "config/stn_gpe/net_qif_syn_adapt",
                       'permute_grid': True,
                       'param_map': param_map,
                       'simulation_time': T,
                       'step_size': dt,
                       'sampling_step_size': dts,
                       'inputs': {},
                       'outputs': {'r_e': "stn/qif_stn/R_e", 'r_i': 'gpe/qif_gpe/R_i'},
                       'init_kwargs': {'backend': 'numpy', 'solver': 'euler', 'step_size': dt},
                   })

    drop_save_dir = f'{compute_dir}/PopulationDrops/'
    os.makedirs(drop_save_dir, exist_ok=True)

    winner = ga.run(
        initial_gene_pool=pop_genes,
        gene_sampling_func=np.random.uniform,
        target=[[20, 60],   # healthy control
                [None, 40],  # stn blockade
                [None, 90],  # gabaa blockade in GPe
                [None, 60],  # STN blockade and GABAA blockade in GPe
                [None, 17],  # blockade of all excitatory inputs to GPe
                [35, 80],  # GABAA antagonist in STN
                [30, 40]   # parkinsonian condition
                ],
        max_iter=1000,
        min_fit=0.95,
        n_winners=6,
        n_parent_pairs=200,
        n_new=50,
        sigma_adapt=0.015,
        candidate_save=f'{compute_dir}/GeneticCGSCandidate.h5',
        drop_save=drop_save_dir,
    )

    winner.to_hdf(f'{compute_dir}/PopulationDrops/winner.h5', key='data')
