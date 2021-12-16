import re
import pathlib
import torch
import torchaudio

from pathlib import Path
from torch.utils.data import Dataset


class CORAADataset(Dataset):

    def __init__(self, data_dir: str, model_sr: int):
        self.audio_files = self._get_audio_files(data_dir)
        self.labels = self._get_labels()
        self.model_sr = model_sr

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple:
        audio_sample = self.audio_files[idx]
        audio_signal = self._get_audio_signal(audio_sample)

        label_sample = self.labels[idx]

        return audio_signal, label_sample

    def _get_audio_files(self, data_dir: str) -> list:
        p = Path(data_dir).glob('**/*')
        audio_files = [str(x.resolve()) for x in p if x.is_file()]
        audio_files = sorted(audio_files)

        return audio_files

    def _get_audio_signal(self, filename):
        signal, sr = torchaudio.load(filename)

        if sr != self.model_sr:
            resampler = torchaudio.transforms.Resample(sr, self.model_sr)
            signal = resampler(signal)

        return signal


    def _get_labels(self) -> list:
        def extract_label_from_filename(filename):
            label = re.search(r".+_(.+).wav", filename).group(1)
            return label

        labels = [extract_label_from_filename(filename) for filename in self.audio_files]
        return labels


if __name__ == "__main__":
    DATA_DIR = "../data/train/"
    dataset = CORAADataset(DATA_DIR, model_sr=16000)

