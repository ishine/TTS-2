import argparse

from g2pk import G2p
import numpy as np
import torch
import yaml

import models
from utils.text import generate_symbol_to_id, text_to_sequence, intersperse


def main(config, ckpt, text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    g2p = G2p()
    phonemes = list(config['phonemes'])
    symbol_to_id = generate_symbol_to_id(phonemes)
    text_seq = g2p(text, descriptive=True, group_vowels=True, to_syl=False)
    text_seq = text_seq.replace("ᄋ", "")
    text_seq = text_to_sequence(text_seq, symbol_to_id)
    if config['add_blank']:
        text_seq = intersperse(text_seq, len(phonemes))
    text_len = torch.tensor(len(text_seq), device=device).unsqueeze(0)
    text_seq = torch.tensor(text_seq, device=device).unsqueeze(0)

    model = getattr(models, config['model'])(config).to(device)
    checkpoint = torch.load(ckpt, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    mel, interpolated_f0 = model.inference(text_seq, text_len)
    np.save(f"{text}_mel.npy", mel)
    if interpolated_f0 is not None:
        np.save(f"{text}_f0.npy", interpolated_f0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str,
                        help='YAML file for configuration')
    parser.add_argument('checkpoint', type=str,
                        help='.pt file')
    parser.add_argument('text', type=str)
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.text)
