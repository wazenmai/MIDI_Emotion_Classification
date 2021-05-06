import pickle
from typing import Callable, Optional
import os
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from ..data import PEmo_Dataset


class PEmoPipeline(LightningDataModule):
    def __init__(self, pipline_config: DictConfig, model_config: DictConfig) -> None:
        super(PEmoPipeline, self).__init__()
        self.model_config = model_config
        self.pipeline_config = pipline_config
        self.dataset_builder = PEmo_Dataset

    def get_fl(self):
        if self.split == "TRAIN":
            self.fl = pd.read_csv("../dataset/split/train.csv", index_col=0)
        elif self.split == "VALID":
            self.fl = pd.read_csv("../dataset/split/val.csv", index_col=0)
        elif self.split == "TEST":
            self.fl = pd.read_csv("../dataset/split/test.csv", index_col=0)
        else:
            print("Split should be one of [TRAIN, VALID, TEST]")


    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = PEmoPipeline.get_dataset(
                self.dataset_builder,
                self.pipeline_config.dataset.feature_path,
                self.pipeline_config.dataset.labels,
                "TRAIN",
                self.pipeline_config.dataset.cls_type,
                self.pipeline_config.type,
                self.pipeline_config.dataset.pad_idx
            )

            self.val_dataset = PEmoPipeline.get_dataset(
                self.dataset_builder,
                self.pipeline_config.dataset.feature_path,
                self.pipeline_config.dataset.labels,
                "VALID",
                self.pipeline_config.dataset.cls_type,
                self.pipeline_config.type,
                self.pipeline_config.dataset.pad_idx
            )

        if stage == "test" or stage is None:
            self.test_dataset = PEmoPipeline.get_dataset(
                self.dataset_builder,
                self.pipeline_config.dataset.feature_path,
                self.pipeline_config.dataset.labels,
                "TEST",
                self.pipeline_config.dataset.cls_type,
                self.pipeline_config.type,
                self.pipeline_config.dataset.pad_idx
            )

    def train_dataloader(self) -> DataLoader:
        return PEmoPipeline.get_dataloader(
            self.train_dataset,
            batch_size=self.pipeline_config.dataloader.params.batch_size,
            num_workers=self.pipeline_config.dataloader.params.num_workers,
            drop_last=False,
            shuffle=True,
            collate_fn = self.train_dataset.batch_padding
        )

    def val_dataloader(self) -> DataLoader:
        return PEmoPipeline.get_dataloader(
            self.val_dataset,
            batch_size=self.pipeline_config.dataloader.params.batch_size,
            num_workers=self.pipeline_config.dataloader.params.num_workers,
            drop_last=False,
            shuffle=False,
            collate_fn= self.val_dataset.batch_padding
        )

    def test_dataloader(self) -> DataLoader:
        return PEmoPipeline.get_dataloader(
            self.test_dataset,
            batch_size=self.pipeline_config.dataloader.params.batch_size,
            num_workers=self.pipeline_config.dataloader.params.num_workers,
            drop_last=False,
            shuffle=False,
            collate_fn= self.test_dataset.batch_padding
        )

    @classmethod
    def get_dataset(cls, dataset_builder: Callable, feature_path, labels, split, cls_type, midi_type, pad_idx) -> Dataset:
        dataset = dataset_builder(feature_path, labels, split, cls_type, midi_type, pad_idx)
        return dataset

    @classmethod
    def get_dataloader(cls, dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool, drop_last: bool, collate_fn, **kwargs) -> DataLoader:
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, drop_last=drop_last, collate_fn=collate_fn, **kwargs)