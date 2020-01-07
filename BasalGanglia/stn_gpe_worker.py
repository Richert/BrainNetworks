# my_cgs_worker.py
from pyrates.utility.grid_search import ClusterWorkerTemplate
import os


class MyWorker(ClusterWorkerTemplate):
    def worker_postprocessing(self, **kwargs):
        for idx, data in self.results.iteritems():
            self.processed_results.loc[:, idx] = data * 1e3
        self.processed_results.index = self.results.index * 1e-3


if __name__ == "__main__":
    cgs_worker = MyWorker()
    #cgs_worker.worker_init()
    cgs_worker.worker_init(
        config_file="/nobackup/spanien1/rgast/PycharmProjects/BrainNetworks/BasalGanglia/results/Config/DefaultConfig_0.yaml",
    subgrid = "/nobackup/spanien1/rgast/PycharmProjects/BrainNetworks/BasalGanglia/results/Grids/Subgrids/DefaultGrid_691/animals/animals_Subgrid_0.h5",
    result_file = "~/my_result.h5",
    build_dir = os.getcwd()
    )
