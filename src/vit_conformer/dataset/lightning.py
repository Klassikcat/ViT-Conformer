from pathlib import Path, PosixPath
import lightning.pytorch as ptl
from typing import *

from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from torch.utils.data import Dataset, DataLoader
from .dataset import AudioDataset


class LightningDataWrapper(ptl.LightningDataModule):
    def __init__(self, train_dataset: Dataset, val_dataset: Dataset, test_dataset: Optional[Dataset]) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    @classmethod
    def from_manifest_files(cls, train_manifest: PosixPath, val_manifest: PosixPath, test_manifest: PosixPath) -> None:
        pass

    @property
    def val_batch_size(self) -> int:
        return self._val_batch_size if self._val_batch_size else self.batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.val_batch_size, shuffle=False)