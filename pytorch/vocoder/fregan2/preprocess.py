import argparse
import json
from pathlib import Path

from joblib import Parallel, delayed
import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from meldataset import mel_spectrogram


def load_audio(path, resample=24000):
    wav, sr = sf.read(path, dtype='float32')
    wav = wav.T
    wav = librosa.to_mono(wav)
    wav = librosa.resample(wav, sr, resample, res_type='scipy')
    return np.clip(wav, -1.0, 1.0)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('data')
    parser.add_argument('config')

    a = parser.parse_args()

    data = a.data
    config = a.config
    with open(config) as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_mels = config["num_mels"]
    num_freq = config["num_freq"]
    n_fft = config["n_fft"]
    hop_size = config["hop_size"]
    win_size = config["win_size"]
    window = config["window"]
    sampling_rate = config["sampling_rate"]
    fmin = config["fmin"]
    fmax = config["fmax"]

    trim_silence = config["trim_silence"]
    trim_threshold_in_db = config["trim_threshold_in_db"]
    trim_frame_size = config["trim_frame_size"]
    trim_hop_size = config["trim_hop_size"]

    dir = f"{data}"
    outdir = f"{dir}_out"
    Path(outdir).mkdir(parents=True, exist_ok=True)
    wavs = list(Path(dir).rglob('*.wav')) + list(Path(dir).rglob('*mic2.flac'))

    def helper(i, path):
        wav = load_audio(path, resample=sampling_rate)
        if trim_silence:
            wav, _ = librosa.effects.trim(wav, trim_threshold_in_db, frame_length=trim_frame_size, hop_length=trim_hop_size)
        y = torch.tensor(wav, device=device).unsqueeze(0)
        mel = mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax).squeeze().cpu().numpy()
        out = f"{outdir}/{i}"
        np.save(f"{out}-wave.npy", wav)
        np.save(f"{out}-feats.npy", mel)

    Parallel(n_jobs=-1, verbose=10)(delayed(helper)(i, path) for i, path in enumerate(wavs))


if __name__ == '__main__':
    main()
