import sys
import os
import logging
import argparse
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='.')
    args = parser.parse_args()

    sys.path.append(args.root)
    try:
        from vit_conformer.lightning import ViTConformerModel, ConformerModel
    except ModuleNotFoundError:
        from src.vit_conformer.lightning import ViTConformerModel, ConformerModel

    model = ViTConformerModel(num_vocab=10)
    base_model = ConformerModel(num_vocab=10)
    transform = MelSpectrogram(n_fft=512, hop_length=256, win_length=512, n_mels=80)
    waveform, _ = torchaudio.load(f'{args.root}/test_data/file_example_WAV_1MG.wav')
    waveform_mono = torch.mean(waveform, dim=0, keepdim=True)
    mel = transform(waveform_mono).transpose(1, 2)  # (B, F, T) -> (B, T, F)

    logging.info(f"size of tensor: {mel.shape}")
    base_model(mel, torch.tensor([len(mel)]))
    model(mel, torch.tensor([100], dtype=torch.float32))
    logging.info("Done!")
#%%
