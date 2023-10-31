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
    def __init__(self, mask_parameter: int):
        super().__init__()
        self.mask_parameter = mask_parameter

    def forward(self, input: Tensor) -> Tensor:
        t = torch.randint(low=0, high=self.mask_parameter, size=(1,)).item()
        t_zero = random.randint(0, input.size(2) - t)



class FrequencyMasking(nn.Module):
    def __init__(self, mask_parameter: int):
        super().__init__()
        self.mask_parameter = mask_parameter

    def forward(self, input: Tensor):
        _, channels, _ = input.shape
        print(input.shape)
        f = torch.randint(low=0, high=self.mask_parameter, size=(1,)).item()
        f_zero = random.choice([i for i in range(0, channels - f)])
        # TODO: add masking for the freuqnecy channel of mel spectrogram given parameters f and f_zero.
        mask = torch.zeros_like(input)
        mask[:, f_zero:f_zero + f, :] = 1
        mask = mask.bool()
        return input.masked_fill(mask, 0)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', type=str, required=True)
    parser.add_argument('--frequency_masking_parameter', type=int, default=27)
    args = parser.parse_args()

    frequency_masking = FrequencyMasking(mask_parameter=args.frequency_masking_parameter)

    audio_file, sample_rate = torchaudio.load(args.audio_path)
    mel_spectrogram = torchaudio.transforms.MelSpectrogram()(audio_file)

    plt.figure(figsize=(12, 4))
    plt.imshow(mel_spectrogram.log2()[0, :, :].numpy())
    plt.show()

    frequency_masked = frequency_masking(mel_spectrogram)

    plt.figure(figsize=(12, 4))
    plt.imshow(frequency_masked.log2()[0, :, :].numpy())
    plt.show()


