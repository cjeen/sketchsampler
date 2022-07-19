import random
from typing import Optional, Sequence

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset


def worker_init_fn(id: int):
    """
    DataLoaders workers init function.

    Initialize the numpy.random seed correctly for each worker, so that
    random augmentations between workers and/or epochs are not identical.

    If a global seed is set, the augmentations are deterministic.

    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)


def collate(batch):
    data = [b for b in zip(*batch)]
    # init_pc, imgs, gt_pc, metadata = data
    imgs, gt_pc, entropy_map, metadata = data

    imgs = torch.from_numpy(np.array(imgs)).requires_grad_(False)
    entropy_map = torch.from_numpy(np.array(entropy_map)).requires_grad_(False)
    gt_pc = [torch.from_numpy(pc).requires_grad_(False) for pc in gt_pc]
    return imgs, gt_pc, entropy_map, metadata


class DataModule(pl.LightningDataModule):
    def __init__(
            self,
            datasets: DictConfig,
            num_workers: DictConfig,
            batch_size: DictConfig,
            cfg: DictConfig,
    ):
        super().__init__()
        self.cfg = cfg

        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.train_dataset: Optional[Dataset] = None
        self.val_datasets: Optional[Sequence[Dataset]] = None
        self.test_datasets: Optional[Sequence[Dataset]] = None

    def prepare_data(self) -> None:
        # download only
        pass

    def setup(self, stage: Optional[str] = None):
        # Here you should instantiate your datasets, you may also split the train into train and validation if needed.
        if stage is None or stage == "fit":
            self.train_dataset = hydra.utils.instantiate(
                self.datasets.train, cfg=self.cfg
            )
            self.val_datasets = [
                hydra.utils.instantiate(dataset_cfg, cfg=self.cfg)
                for dataset_cfg in self.datasets.val
            ]

        if stage is None or stage == "test":
            self.test_datasets = [
                hydra.utils.instantiate(dataset_cfg, cfg=self.cfg)
                for dataset_cfg in self.datasets.test
            ]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size.train,
                          num_workers=self.num_workers.train,
                          collate_fn=collate,
                          worker_init_fn=worker_init_fn, )

    def val_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(dataset,
                       batch_size=self.batch_size.val,
                       num_workers=self.num_workers.val,
                       collate_fn=collate,
                       worker_init_fn=worker_init_fn, )
            for dataset in self.val_datasets
        ]

    def test_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(dataset,
                       batch_size=self.batch_size.test,
                       num_workers=self.num_workers.test,
                       collate_fn=collate,
                       worker_init_fn=worker_init_fn, )
            for dataset in self.test_datasets
        ]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"self.datasets={self.datasets!r}, "
            f"self.num_workers={self.num_workers!r}, "
            f"self.batch_size={self.batch_size!r})"
        )
