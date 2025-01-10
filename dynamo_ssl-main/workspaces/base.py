# from accelerate import Accelerator
from datasets.core import TrajectoryDataset
import torch


class Workspace:
    def __init__(self, cfg, work_dir):
        self.cfg = cfg
        self.work_dir = work_dir
        # self.accelerator = Accelerator()
        self.dataset: TrajectoryDataset = None

    def set_models(self, encoder, projector):
        self.encoder = encoder
        self.projector = projector

    def loader_to_list(self,loader):
        data = []
        for batch in loader:
            for sample in batch:  # 遍历每个样本
                if isinstance(sample, torch.Tensor):
                    # print(f"Sample type: {type(sample)}, Sample content: {sample}")
                    data.append(sample)
        return data

    def set_dataset(self, dataset):
        all_data = []
        all_data.extend(self.loader_to_list(dataset['train_loader']))
        all_data.extend(self.loader_to_list(dataset['test_loader']))
        self.dataset = all_data

    def run_offline_eval(self):
        return {"loss": 0}
