import argparse
from pathlib import Path

import matplotlib
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset import TTSDataset, UniformLengthBatchingSampler, collate_fn
import models
import optim
from optim import Lookahead, RAdam


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
    matplotlib.use('Agg')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    with open(config, encoding='utf8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    training_data = TTSDataset(config['transcript'], config['phonemes'], config['add_blank'])
    validation_data = TTSDataset(config['val_transcript'], config['phonemes'], config['add_blank'])

    batch_size = config['batch_size']
    num_workers = config['num_workers']
    save_and_val_every_n_epoch = config['save_and_val_every_n_epoch']

    training_sampler = UniformLengthBatchingSampler(training_data, batch_size=batch_size)
    training_loader = DataLoader(training_data, batch_sampler=training_sampler, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True, shuffle=False, drop_last=False)

    name = config['name']
    model = getattr(models, config['model'])(config).to(device)

    lr = float(config['lr'])
    betas = config['betas']
    weight_decay = float(config['weight_decay'])
    opt = Lookahead(RAdam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay))


    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda epoch: 1)
    scheduler = config.get('scheduler')
    if scheduler is not None:
        sch = getattr(optim, scheduler['type'])(opt, **scheduler['args'])

    scaler = None
    if config['fp_16']:
        print("Using mixed precision")
        scaler = GradScaler()

    epoch = 0
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
    for e in range(epoch+1, max_epoch+1):
        model.on_train_epoch_start(e)
        for i, batch in enumerate(training_loader):
            batch = [x.to(device) for x in batch]
            opt.zero_grad()
            with autocast(enabled=config['fp_16']):
                loss = model.training_step(batch, i, step, writer)
                writer.add_scalar("lr", sch.get_last_lr()[0], step)
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
                    model.validation_step(batch, i, e, step, writer)
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
