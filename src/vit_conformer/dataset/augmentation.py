import torchaudio
import matplotlib as plt

from typing import Tuple
from pathlib import Path, PosixPath
from omegaconf import DictConfig
from torch import Tensor, nn


class AugmentationModule:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass


class SpecAug(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        pass
