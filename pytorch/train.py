import argparse
from pathlib import Path
import os

import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset import TTSDataset, UniformLengthBatchingSampler, collate_fn
from models.mixertts import MixerTTS
from optim import NoamAnnealing, Lookahead, RAdam


def save(name, step, epoch, model, opt, sch, scaler=None, size=100):
    ckpt = {
        'step': step,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'opt_state_dict': opt.state_dict(),
        'sch_state_dict': sch.state_dict()
    }
    if scaler is not None:
        ckpt['scaler_state_dict'] = scaler.state_dict()
    torch.save(ckpt, f'{name}_{epoch}.pt')
    Path(f'{name}_{epoch - size}.pt').unlink(missing_ok=True)


def main(config, checkpoint):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    with open(config, encoding='utf8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    training_data = TTSDataset(config['transcript'], config['phonemes'])
    validation_data = TTSDataset(config['val_transcript'], config['phonemes'])

    batch_size = config['batch_size']
    num_workers = config['num_workers']
    save_and_val_every_n_epoch = config['save_and_val_every_n_epoch']

    training_sampler = UniformLengthBatchingSampler(training_data, batch_size=batch_size)
    training_loader = DataLoader(training_data, batch_sampler=training_sampler, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True, shuffle=False, drop_last=False)

    name = config['name']
    model = MixerTTS(config).to(device)

    lr = float(config['lr'])
    betas = config['betas']
    weight_decay = float(config['weight_decay'])
    opt = Lookahead(RAdam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay))
    sch = NoamAnnealing(opt, d_model=1, warmup_steps=1000, max_steps=9999999999, min_lr=1e-4)
    scaler = None
    if config['fp_16']:
        print("Using mixed precision")
        scaler = GradScaler()

    epoch = 1
    step = 0
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        opt.load_state_dict(checkpoint['opt_state_dict'])
        sch.load_state_dict(checkpoint['sch_state_dict'])
        epoch = checkpoint['epoch']
        step = checkpoint['step']
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print("Found scaler state dict")
    
    writer = SummaryWriter('runs')

    max_epoch = config['max_epoch']
    for e in range(epoch, max_epoch+1):
        model.on_train_epoch_start(e)
        for i, batch in enumerate(training_loader):
            batch = [x.to(device) for x in batch]
            opt.zero_grad()
            with autocast(enabled=config['fp_16']):
                o = model.training_step(batch, i)
            loss = o['loss']
            progress_bar = o['progress_bar']
            for k, v in progress_bar.items():
                writer.add_scalar(k, v, step)
                writer.add_scalar('lr', sch.get_last_lr()[0], step)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
            else:
                loss.backward()
            loss = torch.nn.utils.clip_grad_norm_(model.parameters(), 1000.0)
            if scaler is not None:
                scaler.step(opt)
                scale = scaler.get_scale()
                scaler.update()
                skip_lr_sched = (scale > scaler.get_scale())
                if not skip_lr_sched:
                    sch.step()
            else:
                opt.step()
                sch.step()
            step += 1
        if e % save_and_val_every_n_epoch == 0:
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(validation_loader):
                    batch = [x.to(device) for x in batch]
                    o = model.validation_step(batch, i, e)
                    specs = o['specs']
                    pitches = o['pitches']
                    alignments = o['alignments']
                    audios = o['audios']
                    for s, p, a1, a2 in zip(specs, pitches, alignments, audios):
                        writer.add_image(s[1], s[0], step, dataformats='HWC')
                        writer.add_image(p[1], p[0], step, dataformats='HWC')
                        writer.add_image(a1[1], a1[0], step, dataformats='HWC')
                        if a2 is not None:
                            writer.add_audio(a2[1], a2[0], step, sample_rate=24000)
                save(name, step, e, model, opt, sch, scaler)
            model.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str,
                        help='YAML file for configuration')
    parser.add_argument('-c', '--checkpoint', type=str,
                        help='.pt file to resume training')
    args = parser.parse_args()
    main(args.config, args.checkpoint)
