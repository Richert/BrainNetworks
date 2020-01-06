# my_cgs_worker.py
import pandas as pd
import numpy as np
from pyrates.utility.grid_search import ClusterWorkerTemplate, grid_search
from pyrates.utility.data_analysis import welch


class MyWorker(ClusterWorkerTemplate):
    def worker_postprocessing(self, **kwargs):
        self.results.index = self.results.index * 1e-3
        self.results = self.results * 1e3


if __name__ == "__main__":
    cgs_worker = MyWorker()
    # cgs_worker.worker_test()
    cgs_worker.worker_init()