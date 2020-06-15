# my_cgs_worker.py
from pyrates.utility.grid_search import ClusterWorkerTemplate
from pandas import DataFrame


class MinimalWorker(ClusterWorkerTemplate):
    def worker_postprocessing(self, **kwargs):
        self.processed_results = DataFrame(data=None, columns=self.results.columns)
        for idx, data in self.results.iteritems():
            self.processed_results.loc[:, idx] = data * 1e3
        self.processed_results.index = self.results.index * 1e-3


if __name__ == "__main__":
    cgs_worker = MinimalWorker()
    cgs_worker.worker_init()
