import random
import torchaudio
import matplotlib.pyplot as plt
import numpy as np

import torch
from typing import Tuple, List
from pathlib import Path, PosixPath
from omegaconf import DictConfig
from torch import Tensor, nn


class AugmentationModule(nn.Module):
    def __init__(self, augment_modules: List[nn.Module]):
        super().__init__()
        self.sequential = nn.Sequential(*augment_modules)

    def __call__(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs)


class SpecAug(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        pass


class TimeWarping(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        pass

    def sparse_image_warp(self, mel_spectrogram: Tensor) -> Tensor:
        pass


class TimeMasking(nn.Module):
    def __init__(self, mask_parameter: int, upper_bound: int, iter: int = 1):
        super().__init__()
        self.mask_parameter = mask_parameter
        self.upper_bound = upper_bound
        self.iter = iter

    def forward(self, input: Tensor) -> Tensor:
        for _ in range(self.iter):
            t = torch.randint(low=0, high=self.mask_parameter, size=(1,)).item()
            if t > self.upper_bound:
                t = self.upper_bound
            total_length = input.shape[2]
            t_zero = random.randint(0, total_length - t)
            mask = torch.zeros_like(input)
            mask[:, :, t_zero:t_zero + t] = 1
            mask = mask.bool()
            input = input.masked_fill(mask, 0)
        return input


class FrequencyMasking(nn.Module):
    def __init__(self, mask_parameter: int, iter: int = 1):
        super().__init__()
        self.mask_parameter = mask_parameter
        self.iter = iter

    def forward(self, input: Tensor):
        _, channels, _ = input.shape
        for _ in range(self.iter):
            f = torch.randint(low=0, high=self.mask_parameter, size=(1,)).item()
            f_zero = random.choice([i for i in range(0, channels - f)])
            # TODO: add masking for the freuqnecy channel of mel spectrogram given parameters f and f_zero.
            mask = torch.zeros_like(input)
            mask[:, f_zero:f_zero + f, :] = 1
            mask = mask.bool()
            input = input.masked_fill(mask, 0)
        return input


