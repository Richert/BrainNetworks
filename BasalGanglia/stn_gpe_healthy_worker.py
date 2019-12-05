# my_cgs_worker.py
import pandas as pd
import numpy as np
from pyrates.utility.grid_search import ClusterWorkerTemplate


class MyWorker(ClusterWorkerTemplate):
    def worker_postprocessing(self, **worker_kwargs):

        self.processed_results = pd.DataFrame(index=['fitness', 'current', 'penal'], columns=self.results.columns.levels[0])

        target = np.array([20, 60])
        tmin = 2.

        for idx, circuit in enumerate(self.result_map.index):
            current = []
            penal = []
            for i, r in enumerate(["r_e", "r_i"]):
                data = self.results.loc[tmin:, (circuit, 'pop', r)]
                if not any(np.isnan(data.values)) and not any(np.isinf(data.values)):
                    current.append(int(np.mean(data)))
                    penal.append(np.abs(np.max(data) - np.min(data)))
                else:
                    current.append(0)
                    penal.append(np.inf)
            fitness = 1 / (1 + np.sum(np.abs(current-target)) + np.sum(penal))
            self.processed_results.loc['fitness', circuit] = fitness
            self.processed_results.loc['current', circuit] = current
            self.processed_results.loc['penal', circuit] = penal


if __name__ == "__main__":
    cgs_worker = MyWorker()
    # cgs_worker.worker_test()
    cgs_worker.worker_init()