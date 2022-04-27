import argparse
from pathlib import Path
import string

from g2pk import G2p
import librosa
import numpy as np

from modules.aligner import beta_binomial_prior_distribution
from utils.audio import load_audio, mel_spectrogram, f0

def main(transcript, strip_o, min_length):
    g2p = G2p()
    phonemes = set()
    trans = str.maketrans('','',string.punctuation)
    root = f"{Path(transcript).parent}"
    with open(f"{transcript}", encoding='utf8') as i, open(f"{transcript}_processed.txt", 'w') as o:
        for line in i:
            path, text = line.strip().split("|")
            text = text.translate(trans)
            text = text.split(" ")
            processed = []
            for sentence in text:
                sentence = g2p(sentence, descriptive=True, group_vowels=True, to_syl=False)
                for p in sentence:
                    phonemes.add(p)
                if strip_o:
                    sentence = sentence.replace("ᄋ", "")
                processed.append(sentence)
            processed = " ".join(processed)
            if len(processed) >= min_length:
                path = f"{root}/{path}"
                wav = load_audio(path)
                wav = librosa.effects.trim(wav, top_db=30, frame_length=1024, hop_length=256)[0]
                mel = mel_spectrogram(wav, 24000)
                f0s = f0(wav)

                mel_len = mel.shape[-1]
                align_prior_matrix = beta_binomial_prior_distribution(len(processed), mel_len)

                path = f"{path}.npz"
                o.write(f"{path}|{processed}\n")
                np.savez(path, mel=mel, f0=f0s, align_prior_matrix=align_prior_matrix)
    print(phonemes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('transcript', type=str)
    parser.add_argument('--strip_o', action='store_true')
    parser.add_argument('--min_length', type=int, default=20)
    args = parser.parse_args()
    main(args.transcript, args.strip_o, args.min_length)
