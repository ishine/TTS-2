import librosa
from librosa.filters import mel as librosa_mel_fn
import numpy as np
import pyworld
import pysptk
import pyreaper
import soundfile as sf
import torch


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_audio(path, resample=24000):
    wav, sr = sf.read(path, dtype='float32')
    wav = wav.T
    wav = librosa.to_mono(wav)
    wav = librosa.resample(wav, sr, resample, res_type='scipy')
    return np.clip(wav, -1.0, 1.0)


def griffin_lim(mel, sr=24000, hop_length=240, n_iter=60, fmin=0, fmax=12000):
    s = librosa.feature.inverse.mel_to_stft(np.exp(mel), sr=sr, n_fft=1024, power=1.0, fmin=fmin, fmax=fmax)
    y = librosa.griffinlim(s, n_iter=n_iter, hop_length=hop_length, win_length=1024, center=True)
    return y


def f0(wav, sr=24000, hop_length=240):
    wav64 = wav.astype(np.float64)
    dio, t = pyworld.dio(wav64, sr, frame_period=10.0, f0_floor=50.0, f0_ceil=1000.0)
    dio = pyworld.stonemask(wav64, dio, t, sr)
    wav16 = wav * 32767
    rapt = pysptk.sptk.rapt(wav16, sr, hop_length, min=50.0, max=1000.0)
    wav16 = wav16.astype(np.int16)
    _, _, _, reaper, _ = pyreaper.reaper(wav16, sr, minf0=50.0, maxf0=1000.0, frame_period=0.01)
    reaper = np.clip(reaper, 0.0, None)
    min_length = min(len(dio), len(rapt), len(reaper))
    stacked = np.stack((dio[:min_length], rapt[:min_length], reaper[:min_length]))
    f0 = np.median(stacked, axis=0)
    return np.pad(f0, (0, len(dio) - min_length), 'edge')


def log_f0(f0s):
    return np.log(np.where(f0s < 1.0, 1.0, f0s)).astype(np.float32)


def excitation(interpolated_f0, sr=24000, sigma=1.0, harmonics=8):
    # https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/blob/master/tutorials/c09_sine_generator.ipynb
    # initial phase noise (no noise for fundamental component)? Not mentioned in paper.
    phase = np.expand_dims(np.random.uniform(-np.pi, np.pi, harmonics), -1)
    h = np.expand_dims(np.arange(1, harmonics+1), -1)
    signal = np.sin(2 * np.pi * (h * np.cumsum(interpolated_f0) / sr) + phase)

    unvoiced = 1 - (np.ones_like(interpolated_f0) * (interpolated_f0 > 0))

    noise = np.random.normal(0, sigma, signal.shape)
    noise = unvoiced * noise
    signal = signal + noise
    return signal


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(wav, sr, n_fft=1024, num_mels=80, win_size=1024, fmin=0, fmax=12000, center=True, hop_size=240, log=True):
    y = torch.tensor(wav, device=device).unsqueeze(0)

    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    if not center:
        y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
        y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    if log:
        spec = spectral_normalize_torch(spec)
    else:
        spec = torch.clamp(spec, min=1e-5)

    return spec.squeeze(0).cpu().numpy()


def mel_normalize(mel, m_min, m_max):
    return 2 * (mel - m_min)/(m_max-m_min) - 1


def mel_denormalize(mel_norm, m_min, m_max):
    return (m_max - m_min) * ((mel_norm)+1)/2 + m_min
