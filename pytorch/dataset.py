import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence

from utils.audio import log_f0
from utils.text import text_to_sequence


def collate_fn(batch):
    text = []
    text_length = []
    mel = []
    mel_length = []
    f0 = []

    for sample in batch:
        text.append(sample['text'])
        text_length.append(sample['text_length'])
        mel.append(sample['mel'].transpose(0, 1))
        mel_length.append(sample['mel_length'])
        f0.append(sample['f0'])

    text = pad_sequence(text, batch_first=True)
    mel = pad_sequence(mel, batch_first=True).transpose(1, 2)
    f0 = pad_sequence(f0, batch_first=True)

    return mel, torch.tensor(mel_length), text, torch.tensor(text_length), f0


class UniformLengthBatchingSampler(Sampler):
    # https://sooftware.io/uniform_length_batching/
    def __init__(self, data_source, batch_size=1):
        super(UniformLengthBatchingSampler, self).__init__(data_source)
        self.data_source = data_source
        #
        data_source.walker = sorted(data_source.walker, key=lambda d: d['text_length'])
        #
        ids = list(range(0, len(data_source)))
        self.bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]
        self.batch_size = batch_size

    def __iter__(self):
        np.random.shuffle(self.bins)
        for ids in self.bins:
            if len(ids) == self.batch_size:
                yield ids

    def __len__(self):
        return len(self.bins)


class TTSDataset(Dataset):
    def __init__(self, transcript):
        self.walker = []
        with open(transcript, encoding='utf8') as f:
            for line in f:
                line = line.strip().split("|")
                text = torch.tensor(text_to_sequence(line[1]))
                data = np.load(line[0])
                mel = torch.tensor(data['mel'])
                log_f0s = torch.tensor(log_f0(data['f0']))
                text_length = len(text)
                mel_length = mel.shape[1]
                f0_length = len(log_f0s)
                data = {'text': text, 'text_length': text_length, 'mel': mel, 'mel_length': mel_length, 'f0': log_f0s, 'f0_length': f0_length}
                self.walker.append(data)
                
    def __getitem__(self, n):
        return self.walker[n]

    def __len__(self):
        return len(self.walker)
