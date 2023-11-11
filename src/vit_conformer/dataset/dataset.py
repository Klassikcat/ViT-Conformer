import ujson
from pathlib import Path, PosixPath
from typing import Tuple, List, Dict, Optional

import torch
import torchaudio
import torchaudio.functional as F
from torch import Tensor, nn
from torch.utils.data import Dataset
from .augmentation import AugmentationModule


class Tokenizer:
    def __init__(self):
        self.convert_token_to_id = {}
        self.convert_id_to_token = {}

    def __len__(self):
        return len(self.convert_token_to_id)

    def tokenize(self, texts_to_tokenize: str) -> List[int]:
        return [self.convert_token_to_id[d] for d in texts_to_tokenize]

    def detokenize(self, ids_to_tokens: List[int]) -> str:
        return ''.join([self.convert_id_to_token[d] for d in ids_to_tokens])

    def convert_texts_to_tokens(self, texts_to_tokenize: str) -> List[int]:
        pass

    def convert_tokens_to_ids(self, tokens_to_convert: List[int]) -> List[int]:
        pass

    def convert_ids_to_tokens(self, ids_to_convert: List[int]) -> List[int]:
        pass

    def convert_tokens_to_string(self, tokens_to_convert: List[int]) -> str:
        pass

    def __call__(self, texts_to_tokenize: str, return_tensors: str) -> List[int]:
        pass


class AudioDataset(Dataset):
    def __init__(
            self,
            datasets: List[Dict[str, str]],
            tokenizer: Tokenizer,
            augment: Optional[AugmentationModule] = None,
            sample_rate: int = 16000,
            num_mels: int = 80,
            f_min: float = 0.0,
            f_max: Optional[float] = None,
            n_fft: int = 512,
            win_length: int = 400,
            hop_length: int = 160,
            pad: int = 0,
            power: int = 2.0,
            normalized: bool = False,
            center: bool = True,
            pad_mode: str = "reflect",
            mel_scale: str = "htk",
    ) -> None:
        """

        :param datasets:
        :param augment:
        :param sample_rate:
        :param num_mels:
        :param f_min:
        :param f_max:
        :param n_fft:
        :param win_length:
        :param hop_length:
        :param pad:
        :param power:
        :param normalized:
        :param center:
        :param pad_mode:
        :param mel_scale:
        """
        super().__init__()
        self.datasets = datasets
        self.augment = augment
        self.tokenizer = tokenizer
        self.transforms = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            pad=pad,
            power=power,
            normalized=normalized,
            center=center,
            pad_mode=pad_mode,
            mel_scale=mel_scale,
            n_mels=num_mels,
            f_min=f_min,
            f_max=f_max,
        )

    def __len__(self):
        return len(self.datasets)

    @classmethod
    def from_folder(cls, folder_path: PosixPath, data_suffix: str = '.wav', label_suffix: str = '.json'):
        """
        Generate Audio dataset from folder path. The folder should have the following structure:
        ```
        folder_path
        ├── audio_1.wav
        ├── audio_2.wav
        ├── audio_3.wav
        ├── ...
        ├── audio_n.wav
        ├── audio_1.json
        ├── audio_2.json
        ├── audio_3.json
        ├── ...
        └── audio_n.json
        ```

        :param folder_path: PosixPath of folder contains datas.
        :param data_suffix: suffix of data files. Default: '.wav'
        :param label_suffix: suffix of label files. Default: '.json'
        :return: AudioDataset
        """
        datasets = []
        for data_path in folder_path.glob(f'*{data_suffix}'):
            label_path = data_path.with_suffix(label_suffix)
            if label_path.exists():
                datasets.append({
                    'data_path': data_path,
                    'label_path': label_path
                })
        return cls(datasets)

    def get_mel_spectrogram(self, audio_tensor: Tensor) -> Tensor:
        return self.transforms(audio_tensor)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        data = self.datasets[idx]
        audio_tensor, sample_rate = torchaudio.load(data['data_path'])
        with open(data['label_path'], 'r') as f:
            label = ujson.load(f)
        mel_spectrogram = self.get_mel_spectrogram(audio_tensor)
        if self.augment:
            mel_spectrogram = self.augment(mel_spectrogram)
        label = self.tokenizer(label, return_tensors='pt')
        return mel_spectrogram, torch.tensor(label)

