import os
import warnings
import pandas as pd
import numpy as np
from pyrates.utility.genetic_algorithm import GSGeneticAlgorithm
from pyrates.utility import grid_search


class CustomGOA(GSGeneticAlgorithm):

    def eval_fitness(self, target: list, **kwargs):
        param_grid = self.pop.drop(['fitness', 'sigma'], axis=1)

        # C1: Resting-state
        ###################

        # perform grid search
        results, param_grid = grid_search(circuit_template=self.gs_config['circuit_template'],
                                          param_grid=param_grid,
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

        # calculate fitness
        for i, gene_id in enumerate(param_grid.index):
            idx = np.argwhere(results.columns.get_level_values(1).values == gene_id).squeeze()
            candidate_out = results.iloc[-1, idx].values
            dist = self.fitness_measure(candidate_out, target, **self.fitness_kwargs)
            self.pop.at[i, 'fitness'] = float(1 / dist)


def fitness(y, t):
    diff = y - t
    return np.sqrt(np.mean(diff**2))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    pop_genes = {
        'k_ee': {
            'min': 0,
            'max': 20,
            'N': 2,
            'sigma': 0.4
        },
        'k_ei': {
            'min': 10,
            'max': 200,
            'N': 4,
            'sigma': 0.8
        },
        'k_ie': {
            'min': 10,
            'max': 200,
            'N': 4,
            'sigma': 0.8
        },
        'k_ii': {
            'min': 0,
            'max': 100,
            'N': 2,
            'sigma': 0.8
        },
        'eta_e': {
            'min': -10,
            'max': 10,
            'N': 2,
            'sigma': 0.4
        },
        'eta_i': {
            'min': -10,
            'max': 10,
            'N': 2,
            'sigma': 0.4
        },
        'alpha': {
            'min': 0,
            'max': 10.0,
            'N': 2,
            'sigma': 0.2
        }
    }

    param_map = {
        'k_ee': {'vars': ['qif_stn/k_ee'],
                 'nodes': ['stn']},
        'k_ei': {'vars': ['qif_stn/k_ei'],
                 'nodes': ['stn']},
        'k_ie': {'vars': ['qif_gpe/k_ie'],
                 'nodes': ['gpe']},
        'k_ii': {'vars': ['qif_gpe/k_ii'],
                 'nodes': ['gpe']},
        'eta_e': {'vars': ['qif_stn/eta_e'],
                  'nodes': ['stn']},
        'eta_i': {'vars': ['qif_gpe/eta_i'],
                  'nodes': ['gpe']},
        'alpha': {'vars': ['qif_gpe/alpha'],
                  'nodes': ['gpe']}
    }

    T = 1.
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
                       'init_kwargs': {'backend': 'numpy', 'solver': 'scipy', 'step_size': dt}
                   })

    drop_save_dir = f'{compute_dir}/PopulationDrops/'
    os.makedirs(drop_save_dir, exist_ok=True)

    winner = ga.run(
        initial_gene_pool=pop_genes,
        gene_sampling_func=np.random.uniform,
        target=[20, 60],
        max_iter=1000,
        min_fit=0.95,
        n_winners=12,
        n_parent_pairs=400,
        n_new=100,
        sigma_adapt=0.015,
        candidate_save=f'{compute_dir}/GeneticCGSCandidate.h5',
        drop_save=drop_save_dir,
        method='RK23'
    )

    winner.to_hdf(f'{compute_dir}/PopulationDrops/winner.h5', key='data')
