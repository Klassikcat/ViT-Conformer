import torchaudio
import matplotlib.pyplot as plt
try:
    from vit_conformer.dataset.augmentation import FrequencyMasking, TimeMasking, TimeWarping
except ImportError:
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.vit_conformer.dataset.augmentation import FrequencyMasking, TimeMasking, TimeWarping


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', type=str, required=True)
    parser.add_argument('--frequency_masking_parameter', type=int, default=27)
    parser.add_argument('--frequency_masking_iter', type=int, default=2)
    parser.add_argument('--time_masking_parameter', type=int, default=27)
    parser.add_argument('--time_masking_upper_bound', type=int, default=100)
    parser.add_argument('--time_masking_iter', type=int, default=2)
    args = parser.parse_args()

    frequency_masking = FrequencyMasking(mask_parameter=args.frequency_masking_parameter, iter=args.frequency_masking_iter)
    time_masking = TimeMasking(mask_parameter=args.time_masking_parameter, upper_bound=args.time_masking_upper_bound, iter=args.time_masking_iter)

    audio_file, sample_rate = torchaudio.load(args.audio_path)
    mel_spectrogram = torchaudio.transforms.MelSpectrogram()(audio_file)

    plt.figure(figsize=(12, 4))
    plt.imshow(mel_spectrogram.log2()[0, :, :].numpy(), cmap='plasma')
    plt.show()

    frequency_masked = frequency_masking(mel_spectrogram)

    plt.figure(figsize=(12, 4))
    plt.imshow(frequency_masked.log2()[0, :, :].numpy(), cmap='plasma')
    plt.show()

    time_masked = time_masking(mel_spectrogram)

    plt.figure(figsize=(12, 4))
    plt.imshow(time_masked.log2()[0, :, :].numpy(), cmap='plasma')
    plt.show()

    mask_applied = time_masking(frequency_masking(time_masked))

    plt.figure(figsize=(12, 4))
    plt.imshow(mel_spectrogram.log2()[0, :, :].numpy(), cmap='plasma')
    plt.show()



