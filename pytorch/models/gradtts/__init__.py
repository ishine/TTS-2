# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import math
import random

import numpy as np
import torch
import torch.nn as nn

from monotonic_align import maximum_path
from .text_encoder import TextEncoder
#from .unet import Diffusion
from .wavenet import Decoder
from .util import sequence_mask, generate_path, duration_loss, fix_len_compatibility
from utils import plot_alignment_to_numpy, plot_pitch_to_numpy, plot_spectrogram_to_numpy, regulate_len, average_pitch
from utils.audio import griffin_lim, mel_denormalize, mel_normalize
from modules.aligner import bin_loss, forward_sum_loss


class GradTTS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_vocab = config.get('n_vocab')
        if config.get('add_blank'):
            self.n_vocab += 1
        self.n_spks = config.get('n_spks')
        self.spk_emb_dim = config.get('spk_emb_dim')
        self.n_enc_channels = config.get('n_enc_channels')
        self.filter_channels = config.get('filter_channels')
        self.filter_channels_dp = config.get('filter_channels_dp')
        self.n_heads = config.get('n_heads')
        self.n_enc_layers = config.get('n_enc_layers')
        self.enc_kernel = config.get('enc_kernel')
        self.enc_dropout = config.get('enc_dropout')
        self.window_size = config.get('window_size')
        self.n_feats = config.get('n_feats')
        self.beta_min = config.get('beta_min')
        self.beta_max = config.get('beta_max')
        self.pe_scale = config.get('pe_scale')
        self.pitch = config.get('pitch')
        self.n_layers = config.get('n_layers')
        self.dec_dim = config.get('dec_dim')
        self.max_steps = config.get('max_steps')
        self.loss_fn = nn.L1Loss() if config.get('loss_fn') == 'MAE' else nn.MSELoss()
        self.mel_min = config.get('mel_min')
        self.mel_max = config.get('mel_max')
        self.pitch = config.get('pitch')

        if self.n_spks > 1:
            self.spk_emb = nn.Embedding(self.n_spks, self.spk_emb_dim)
        self.encoder = TextEncoder(self.n_vocab, self.n_feats, self.n_enc_channels, 
                                   self.filter_channels, self.filter_channels_dp, self.n_heads, 
                                   self.n_enc_layers, self.enc_kernel, self.enc_dropout, self.window_size,
                                   pitch=self.pitch)
        self.decoder = Decoder(self.n_feats, self.n_enc_channels, self.n_layers, self.dec_dim, self.max_steps, self.loss_fn)

    def relocate_input(self, x: list):
        """
        Relocates provided tensors to the same device set for the module.
        """
        device = next(self.parameters()).device
        for i in range(len(x)):
            if isinstance(x[i], torch.Tensor) and x[i].device != device:
                x[i] = x[i].to(device)
        return x

    @torch.no_grad()
    def forward(self, x, x_lengths, n_timesteps, temperature=1.0, stoc=False, spk=None, length_scale=1.0,
                durations=None, pitch=None):
        """
        Generates mel-spectrogram from text. Returns:
            1. encoder outputs
            2. decoder outputs
            3. generated alignment
        
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            stoc (bool, optional): flag that adds stochastic term to the decoder sampler.
                Usually, does not provide synthesis improvements.
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.
            duration (list[int], optional): list of lenghts of texts.
                Use this instead of prediction durations.
            pitch (list[float], optional): list of f0s of texts.
                Use this instead of predicted pitches.
        """
        x, x_lengths = self.relocate_input([x, x_lengths])

        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(spk)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask, pitch, h, attn_soft, attn_logprob, attn_hard, attn_hard_dur = self.encoder(x, x_lengths, pitch=None, spk=spk)
        if durations is not None:
            logw = durations

        if not self.pitch:
            w = torch.exp(logw) * x_mask
            w_ceil = torch.ceil(w) * length_scale
            y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
            y_max_length = int(y_lengths.max())
            #y_max_length_ = fix_len_compatibility(y_max_length)

            # Using obtained durations `w` construct alignment map `attn`
            y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask.dtype)
            attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
            attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

            # Align encoded text and get mu_y
            mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), h.transpose(1, 2))
            mu_y = mu_y.transpose(1, 2)
            encoder_outputs = mu_y[:, :, :y_max_length]
            attn = attn[:, :, :y_max_length]
        else:
            durs_predicted = torch.clamp(logw.exp() - 1, 0) * x_mask
            durs_predicted = durs_predicted.squeeze(1)
            mu_y, y_lengths = regulate_len(durs_predicted, h.transpose(1, 2))
            mu_y = mu_y.transpose(1, 2)
            y_max_length = int(y_lengths.max())
            #y_max_length_ = fix_len_compatibility(y_max_length)
            y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask.dtype)
            encoder_outputs = mu_y[:, :, :y_max_length]
            attn = None
            pitch, _ = regulate_len(durs_predicted, pitch.transpose(1, 2))
            pitch = pitch.squeeze(2)
            pitch = pitch[:, :y_max_length]

        # Sample latent representation from terminal distribution N(mu_y, I)
        # z = mu_y + torch.randn_like(mu_y, device=mu_y.device) / temperature
        z = torch.randn(mu_y.shape[0], self.n_feats, mu_y.shape[2], device=mu_y.device) / temperature
        # Generate sample by performing reverse dynamics
        decoder_outputs = self.decoder(z, y_mask, mu_y, n_timesteps, stoc, spk)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]
        # denormalize mel
        decoder_outputs = mel_denormalize(decoder_outputs, self.mel_min, self.mel_max)

        return encoder_outputs, decoder_outputs, attn, pitch, y_lengths

    def compute_loss(self, x, x_lengths, y, y_lengths, attn_prior=None, pitch=None, spk=None, out_size=None):
        """
        Computes 3 losses:
            1. duration loss: loss between predicted token durations and those extracted by Monotinic Alignment Search (MAS).
            2. prior loss: loss between mel-spectrogram and encoder outputs.
            3. diffusion loss: loss between gaussian noise and its reconstruction by diffusion-based decoder.
            
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            y (torch.Tensor): batch of corresponding mel-spectrograms.
            y_lengths (torch.Tensor): lengths of mel-spectrograms in batch.
            out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
        """
        # x, x_lengths, y, y_lengths = self.relocate_input([x, x_lengths, y, y_lengths])

        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(spk)
        
        # normalize mel
        y = mel_normalize(y, self.mel_min, self.mel_max)
        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask, predicted_pitch, h, attn_soft, attn_logprob, attn_hard, attn_hard_dur = self.encoder(x, x_lengths, y=y, y_lengths=y_lengths, attn_prior=attn_prior, pitch=pitch, spk=spk)
        y_max_length = y.shape[-1]

        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)

        if not self.pitch:
            attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

            # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
            with torch.no_grad(): 
                const = -0.5 * math.log(2 * math.pi) * self.n_feats
                factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
                y_square = torch.matmul(factor.transpose(1, 2), y ** 2)
                y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
                mu_square = torch.sum(factor * (mu_x ** 2), 1).unsqueeze(-1)
                log_prior = y_square - y_mu_double + mu_square + const

                attn = maximum_path(log_prior, attn_mask.squeeze(1))
                attn = attn.detach()

            # Compute loss between predicted log-scaled durations and those obtained from MAS
            logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
            dur_loss = duration_loss(logw, logw_, x_lengths)

            # Cut a small segment of mel-spectrogram in order to increase batch size
            if not isinstance(out_size, type(None)):
                max_offset = (y_lengths - out_size).clamp(0)
                offset_ranges = list(zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
                out_offset = torch.LongTensor([
                    torch.tensor(random.choice(range(start, end)) if end > start else 0)
                    for start, end in offset_ranges
                ]).to(y_lengths)
                
                attn_cut = torch.zeros(attn.shape[0], attn.shape[1], out_size, dtype=attn.dtype, device=attn.device)
                y_cut = torch.zeros(y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device)
                y_cut_lengths = []
                for i, (y_, out_offset_) in enumerate(zip(y, out_offset)):
                    y_cut_length = out_size + (y_lengths[i] - out_size).clamp(None, 0)
                    y_cut_lengths.append(y_cut_length)
                    cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                    y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                    attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]
                y_cut_lengths = torch.LongTensor(y_cut_lengths)
                y_cut_mask = sequence_mask(y_cut_lengths).unsqueeze(1).to(y_mask)
                
                attn = attn_cut
                y = y_cut
                y_mask = y_cut_mask

            # Align encoded text with mel-spectrogram and get mu_y segment
            mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), h.transpose(1, 2))
            mu_y = mu_y.transpose(1, 2)
            mel_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
            mel_y = mel_y.transpose(1, 2)

            # Compute loss of score-based decoder
            diff_loss, xt = self.decoder.compute_loss(y, y_mask, mu_y, spk)
            
            # Compute loss between aligned encoder outputs and mel-spectrogram
            prior_loss = 0
            prior_loss = torch.sum(0.5 * ((y - mel_y) ** 2 + math.log(2 * math.pi)) * y_mask)
            prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)
            
            return dur_loss, prior_loss, diff_loss
        else:
            len_regulated_enc_hidden, _ = regulate_len(attn_hard_dur, h.transpose(1, 2))
            len_regulated_mel_proj, _ = regulate_len(attn_hard_dur, mu_x. transpose(1, 2))

            true_durs = attn_hard_dur
            pred_durs = logw.squeeze(1)

            text_mask = sequence_mask(x_lengths)
            dur_loss = nn.functional.mse_loss(pred_durs, (true_durs + 1).float().log(), reduction='none')
            dur_loss = dur_loss * text_mask.float()
            dur_loss = dur_loss.sum() / text_mask.sum()

            ctc_loss = forward_sum_loss(attn_logprob=attn_logprob, in_lens=x_lengths, out_lens=y_lengths)
            b_loss = bin_loss(hard_attention=attn_hard, soft_attention=attn_soft)

            true_avg_pitch = average_pitch(pitch.unsqueeze(1), attn_hard_dur).squeeze(1)
            pitch_loss = nn.functional.mse_loss(predicted_pitch.squeeze(1), true_avg_pitch, reduction='none')  # noqa
            pitch_loss = (pitch_loss * text_mask).sum() / text_mask.sum()

            # Compute loss of score-based decoder
            diff_loss, xt = self.decoder.compute_loss(y, y_mask, len_regulated_enc_hidden.transpose(1, 2), spk)
            
            # Compute loss between aligned encoder outputs and mel-spectrogram
            prior_loss = 0
            prior_loss = torch.sum(0.5 * ((y - len_regulated_mel_proj.transpose(1, 2)) ** 2 + math.log(2 * math.pi)) * y_mask)
            prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)
            prior_loss = 0

            return dur_loss, prior_loss, diff_loss, pitch_loss, ctc_loss, b_loss

    def on_train_epoch_start(self, epoch):
        return

    def training_step(self, batch, batch_idx, step, writer):
        spect, spect_len, text, text_len, pitch, attn_prior = batch
        loss = 0
        if not self.pitch:
            dur_loss, prior_loss, diff_loss = self.compute_loss(text, text_len, spect, spect_len, attn_prior, pitch)
        else:
            dur_loss, prior_loss, diff_loss, pitch_loss, ctc_loss, b_loss = self.compute_loss(text, text_len, spect, spect_len, attn_prior, pitch)
            writer.add_scalar("Forward sum loss", ctc_loss, step)
            writer.add_scalar("Bin loss", b_loss, step)
            writer.add_scalar("Pitch loss", pitch_loss, step)
            loss = loss + pitch_loss + ctc_loss + b_loss
        writer.add_scalar("Duration loss", dur_loss, step)
        writer.add_scalar("Diffusion loss", diff_loss, step)
        if prior_loss != 0:
            writer.add_scalar("Prior loss", prior_loss, step)
        loss = loss + dur_loss + prior_loss + diff_loss
        return loss

    def validation_step(self, batch, batch_idx, epoch, step, writer):
        spect, spect_len, text, text_len, pitch, attn_prior = batch
        encoder_outputs, decoder_outputs, attn, predicted_pitch, y_lengths = self(text, text_len, -1, pitch=pitch)
        if batch_idx == 0:
            for i in range(min(3, spect.shape[0])):
                writer.add_image(f"gt mel {i}", plot_spectrogram_to_numpy(spect[i, :, : spect_len[i]].data.cpu().numpy()), step, dataformats='HWC')
                writer.add_image(f"pred mel {i}", plot_spectrogram_to_numpy(decoder_outputs[i, :, : y_lengths[i]].data.cpu().numpy()), step, dataformats='HWC')
                if attn:
                    writer.add_image(f"alignment {i}", plot_alignment_to_numpy(attn[i].data.cpu().numpy().squeeze().T), step, dataformats='HWC')
                if epoch > 100:
                    writer.add_audio(f"gt audio {i}", np.expand_dims(griffin_lim(spect[i, :, : spect_len[i]].data.cpu().numpy()), axis=0), step, sample_rate=24000)
                    writer.add_audio(f"pred audio {i}", np.expand_dims(griffin_lim(decoder_outputs[i, :, : y_lengths[i]].data.cpu().numpy()), axis=0), step, sample_rate=24000)
                if self.pitch:
                    writer.add_image(f"gt pitch {i}", plot_pitch_to_numpy(pitch[i, : spect_len[i]].data.cpu().numpy()), step, dataformats='HWC')
                    writer.add_image(f"pred pitch {i}", plot_pitch_to_numpy(predicted_pitch[i, : y_lengths[i]].data.cpu().numpy()), step, dataformats='HWC')
        