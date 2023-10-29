from pathlib import Path, PosixPath
from typing import Tuple, List

import torch
import torchaudio
from torch import Tensor, nn
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self, dataset_path: PosixPath, data_suffix: str, label_suffix: str) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.data_suffix = data_suffix
        self.label_suffix = label_suffix

    def __len__(self):
        pass

    @property
    def data_fullpath(self) :
        return self.dataset_path / f'*.{self.data_suffix}'

    @property
    def label_fullpath(self) -> Path:
        return self.dataset_path / f'*.{self.label_suffix}'

    def get_spectrogram(self, audio_tensor: Tensor) -> Tensor:
        pass

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        pass

