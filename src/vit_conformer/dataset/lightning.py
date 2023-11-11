from copy import deepcopy
import torch.nn as nn
from torch.utils.data import ConcatDataset

from pathlib import Path, PosixPath
import lightning.pytorch as ptl
from typing import *

from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from torch.utils.data import Dataset, DataLoader
from .dataset import AudioDataset
from .augmentation import AugmentationModule


class LightningDataWrapper(ptl.LightningDataModule):
    def __init__(
            self,
            train_dataset: Dataset,
            val_dataset: Dataset,
            test_dataset: Optional[Dataset],
            augment_module: List[AugmentationModule],
            batch_size: int,
            val_batch_size: Optional[int] = None
    ) -> None:
        super().__init__()
        self._train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.augment_module = nn.ModuleList(augment_module)
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size

    @property
    def train_dataset(self) -> Dataset:
        if not self.augment_module:
            return self._train_dataset
        new_module = deepcopy(self._train_dataset)
        new_module.augment_module = self.augment_module
        return ConcatDataset([self._train_dataset, new_module])

    @classmethod
    def from_manifest_files(
            cls,
            train_manifest: PosixPath,
            val_manifest: PosixPath,
            test_manifest: PosixPath,
            augment_module: List[AugmentationModule],
            batch_size: int,
            val_batch_size: Optional[int] = None
    ) -> "LightningDataWrapper":
        train_dataset = AudioDataset(
            dataset_path=train_manifest.parent,
            data_suffix='wav',
            label_suffix='txt'
        )
        val_dataset = AudioDataset(val_manifest)
        test_dataset = AudioDataset(test_manifest)
        return cls(
            train_dataset,
            val_dataset,
            test_dataset,
            augment_module,
            batch_size=batch_size,
            val_batch_size=val_batch_size
        )

    @property
    def val_batch_size(self) -> int:
        return self.val_batch_size if self.val_batch_size else self.batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.val_batch_size, shuffle=False)
