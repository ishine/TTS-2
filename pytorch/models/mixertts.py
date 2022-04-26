import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from modules.aligner import binarize_attention_parallel, bin_loss, forward_sum_loss, AlignmentEncoder
from utils import get_mask_from_lengths, average_pitch, regulate_len, plot_pitch_to_numpy, plot_spectrogram_to_numpy
from utils.audio import griffin_lim


class ChannelMixBlock(nn.Module):
    def __init__(self, features, expansion_factor, dropout):
        super().__init__()
        self.linear1 = nn.Linear(features, features * expansion_factor)
        self.linear2 = nn.Linear(features * expansion_factor, features)
        self.dropout = dropout

    def forward(self, x, mask):
        x = self.linear1(x) * mask
        x = F.gelu(x)
        x = F.dropout(x, self.dropout)
        x = self.linear2(x) * mask
        x = F.dropout(x, self.dropout)
        return x


class TimeMixBlock(nn.Module):
    def __init__(self, features, kernel_size, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(features, features, kernel_size, padding=(kernel_size - 1) // 2, groups=features)
        self.conv2 = nn.Conv1d(features, features, kernel_size, padding=(kernel_size - 1) // 2, groups=features)
        self.dropout = dropout

    def forward(self, x, mask):
        x = self.conv1(x.transpose(1, 2)).transpose(1, 2) * mask
        x = F.gelu(x)
        x = F.dropout(x, self.dropout)
        x = self.conv2(x.transpose(1, 2)).transpose(1, 2) * mask
        x = F.dropout(x, self.dropout)
        return x


class MixerTTSBlock(nn.Module):
    def __init__(self, features, kernel_size, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(features)
        self.mix1 = TimeMixBlock(features, kernel_size, dropout)
        self.norm2 = nn.LayerNorm(features)
        self.mix2 = ChannelMixBlock(features, kernel_size, dropout)

    def forward(self, x, mask):
        o = self.norm1(x)
        o = self.mix1(o, mask)
        x = x + o
        o = self.norm2(x)
        o = self.mix2(o, mask)
        x = x + o
        return x # B x L x C


class Predictor(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2)
        self.norm1 = nn.LayerNorm(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2)
        self.norm2 = nn.LayerNorm(out_channels)
        self.linear = nn.Linear(out_channels, 1)
        self.dropout = dropout

    def forward(self, x, mask):
        x = x * mask # B x L x C
        x = x.transpose(1, 2) # B x C x L
        x = self.conv1(x)
        x = F.relu(x)
        x = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        x = F.dropout(x, self.dropout)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.norm2(x.transpose(1, 2)).transpose(1, 2)
        x = F.dropout(x, self.dropout)
        x = x.transpose(1, 2) # B x L x C
        x = self.linear(x)
        x = x * mask
        return x.squeeze(-1) # B x L


class MixerTTS(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        num_embeddings = config['num_embeddings']
        embedding_dim = config['embedding_dim']
        encoder_kernels = config['encoder_kernels']
        decoder_kernels = config['decoder_kernels']

        self.pitch_loss_scale = config['pitch_loss_scale']
        self.durs_loss_scale = config['durs_loss_scale']
        self.mel_loss_scale = config['mel_loss_scale']

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.encoder = nn.ModuleList()
        for k in encoder_kernels:
            self.encoder.append(MixerTTSBlock(embedding_dim, k, 0.15))
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.pitch_predictor = Predictor(embedding_dim, 256, 3, 0.15)
        self.pitch_emb = nn.Conv1d(1, embedding_dim, 3, padding=1)
        self.duration_predictor = Predictor(embedding_dim, 256, 3, 0.15)
        self.decoder = nn.ModuleList()
        for k in decoder_kernels:
            self.decoder.append(MixerTTSBlock(embedding_dim, k, 0.15))
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.linear = nn.Linear(embedding_dim, 80)

        self.aligner = AlignmentEncoder(n_text_channels=embedding_dim)

        self.add_bin_loss = False
        self.bin_loss_scale = 1.0
        self.bin_loss_start_epoch = config['bin_loss_start_epoch']


    def align(self, text, text_len, text_mask, spect, spect_len, attn_prior):
        text_emb = self.embedding(text)
        attn_soft, attn_logprob = self.aligner(
            spect, text_emb.permute(0, 2, 1), mask=text_mask == 0, attn_prior=attn_prior,
        )
        attn_hard = binarize_attention_parallel(attn_soft, text_len, spect_len)
        attn_hard_dur = attn_hard.sum(2)[:, 0, :]
        assert torch.all(torch.eq(attn_hard_dur.sum(dim=1), spect_len))
        return attn_soft, attn_logprob, attn_hard, attn_hard_dur

    def _metrics(
        self,
        true_durs,
        true_text_len,
        pred_durs,
        true_pitch,
        pred_pitch,
        true_spect=None,
        pred_spect=None,
        true_spect_len=None,
        attn_logprob=None,
        attn_soft=None,
        attn_hard=None,
        attn_hard_dur=None,
    ):
        text_mask = get_mask_from_lengths(true_text_len)
        mel_mask = get_mask_from_lengths(true_spect_len)
        loss = 0.0

        # Dur loss and metrics
        durs_loss = F.mse_loss(pred_durs, (true_durs + 1).float().log(), reduction='none')
        durs_loss = durs_loss * text_mask.float()
        durs_loss = durs_loss.sum() / text_mask.sum()

        durs_pred = pred_durs.exp() - 1
        durs_pred = torch.clamp_min(durs_pred, min=0)
        durs_pred = durs_pred.round().long()

        acc = ((true_durs == durs_pred) * text_mask).sum().float() / text_mask.sum() * 100
        acc_dist_1 = (((true_durs - durs_pred).abs() <= 1) * text_mask).sum().float() / text_mask.sum() * 100
        acc_dist_3 = (((true_durs - durs_pred).abs() <= 3) * text_mask).sum().float() / text_mask.sum() * 100

        pred_spect = pred_spect.transpose(1, 2)

        # Mel loss
        mel_loss = F.mse_loss(pred_spect, true_spect, reduction='none').mean(dim=-2)
        mel_loss = mel_loss * mel_mask.float()
        mel_loss = mel_loss.sum() / mel_mask.sum()

        loss = loss + self.durs_loss_scale * durs_loss + self.mel_loss_scale * mel_loss

        # Aligner loss
        b_loss, ctc_loss = None, None
        ctc_loss = forward_sum_loss(attn_logprob=attn_logprob, in_lens=true_text_len, out_lens=true_spect_len)
        loss = loss + ctc_loss
        if self.add_bin_loss:
            b_loss = bin_loss(hard_attention=attn_hard, soft_attention=attn_soft)
            loss = loss + self.bin_loss_scale * b_loss
        true_avg_pitch = average_pitch(true_pitch.unsqueeze(1), attn_hard_dur).squeeze(1)

        # Pitch loss
        pitch_loss = F.mse_loss(pred_pitch, true_avg_pitch, reduction='none')  # noqa
        pitch_loss = (pitch_loss * text_mask).sum() / text_mask.sum()

        loss = loss + self.pitch_loss_scale * pitch_loss

        return loss, durs_loss, acc, acc_dist_1, acc_dist_3, pitch_loss, mel_loss, ctc_loss, b_loss

    def forward(self, text, text_len, pitch=None, spect=None, spect_len=None, attn_prior=None):
        if self.training:
            assert pitch is not None
        
        text_mask = get_mask_from_lengths(text_len).unsqueeze(2) # B x L x 1

        text_emb = self.embedding(text) # B x L x C
        h = text_emb * text_mask # B x L x C
        for b in self.encoder:
            h = b(h, text_mask)
        enc_out = self.norm1(h) # B x L x C

        attn_soft, attn_logprob, attn_hard, attn_hard_dur = None, None, None, None
        if spect is not None:
            attn_soft, attn_logprob, attn_hard, attn_hard_dur = self.align(
                text, text_len, text_mask, spect, spect_len, attn_prior
            )

        log_durs_predicted = self.duration_predictor(enc_out, text_mask)
        durs_predicted = torch.clamp(log_durs_predicted.exp() - 1, 0)

        pitch_predicted = self.pitch_predictor(enc_out, text_mask)

        if not self.training:
            if pitch is not None:
                pitch = average_pitch(pitch.unsqueeze(1), attn_hard_dur).squeeze(1)
                pitch_emb = self.pitch_emb(pitch.unsqueeze(1))
            else:
                pitch_emb = self.pitch_emb(pitch_predicted.unsqueeze(1))
        else:
            pitch = average_pitch(pitch.unsqueeze(1), attn_hard_dur).squeeze(1)
            pitch_emb = self.pitch_emb(pitch.unsqueeze(1))

        enc_out = enc_out + pitch_emb.transpose(1, 2) # B x L x C

        len_regulated_enc_out, dec_lens = regulate_len(attn_hard_dur, enc_out)

        dec_mask = get_mask_from_lengths(dec_lens).unsqueeze(2)

        h = len_regulated_enc_out * dec_mask
        for b in self.decoder:
            h = b(h, dec_mask)
        pred_spect = self.linear(h) # B x L x C
        return (
            pred_spect,
            durs_predicted,
            log_durs_predicted,
            pitch_predicted,
            attn_soft,
            attn_logprob,
            attn_hard,
            attn_hard_dur,
        )

    def infer(
        self,
        text,
        text_len=None,
        text_mask=None,
        spect=None,
        spect_len=None,
        attn_prior=None,
        use_gt_durs=False,
        pitch=None,
    ):
        if text_mask is None:
            text_mask = get_mask_from_lengths(text_len).unsqueeze(2)

        text_emb = self.embedding(text)
        h = text_emb * text_mask
        for b in self.encoder:
            h = b(h, text_mask)
        enc_out = self.norm1(h)

        # Aligner
        attn_hard_dur = None
        if use_gt_durs:
            attn_soft, attn_logprob, attn_hard, attn_hard_dur = self.align(
                text, text_len, text_mask, spect, spect_len, attn_prior
            )

        # Duration predictor
        log_durs_predicted = self.duration_predictor(enc_out, text_mask)
        durs_predicted = torch.clamp(log_durs_predicted.exp() - 1, 0)

        # Avg pitch, pitch predictor
        if use_gt_durs and pitch is not None:
            pitch = average_pitch(pitch.unsqueeze(1), attn_hard_dur).squeeze(1)
            pitch_emb = self.pitch_emb(pitch.unsqueeze(1))
        else:
            pitch_predicted = self.pitch_predictor(enc_out, text_mask)
            pitch_emb = self.pitch_emb(pitch_predicted.unsqueeze(1))

        # Add pitch emb
        enc_out = enc_out + pitch_emb.transpose(1, 2)

        if use_gt_durs:
            if attn_hard_dur is not None:
                len_regulated_enc_out, dec_lens = regulate_len(attn_hard_dur, enc_out)
            else:
                raise NotImplementedError
        else:
            len_regulated_enc_out, dec_lens = regulate_len(durs_predicted, enc_out)

        dec_mask = get_mask_from_lengths(dec_lens).unsqueeze(2)

        h = len_regulated_enc_out * dec_mask
        for b in self.decoder:
            h = b(h, dec_mask)
        pred_spect = self.linear(h)

        return pred_spect # B x L x C

    def on_train_epoch_start(self, epoch):
        #bin_loss_start_epoch = np.ceil(self.bin_loss_start_ratio * self.max_epoch)

        # Add bin loss when current_epoch >= bin_start_epoch
        if not self.add_bin_loss and epoch >= self.bin_loss_start_epoch:
            print(f"Using hard attentions after epoch: {self.current_epoch}")
            self.add_bin_loss = True

        #if self.add_bin_loss:
        #    self.bin_loss_scale = min((epoch - bin_loss_start_epoch) / self.bin_loss_warmup_epochs, 1.0)

    def training_step(self, batch, batch_idx):
        attn_prior = None
        spect, spect_len, text, text_len, pitch = batch

        # pitch normalization
        # zero_pitch_idx = pitch == 0
        # pitch = (pitch - self.pitch_mean) / self.pitch_std
        # pitch[zero_pitch_idx] = 0.0

        (pred_spect, _, pred_log_durs, pred_pitch, attn_soft, attn_logprob, attn_hard, attn_hard_dur,) = self(
            text=text,
            text_len=text_len,
            pitch=pitch,
            spect=spect,
            spect_len=spect_len,
            attn_prior=attn_prior
        )

        (loss, durs_loss, acc, acc_dist_1, acc_dist_3, pitch_loss, mel_loss, ctc_loss, bin_loss,) = self._metrics(
            pred_durs=pred_log_durs,
            pred_pitch=pred_pitch,
            true_durs=attn_hard_dur,
            true_text_len=text_len,
            true_pitch=pitch,
            true_spect=spect,
            pred_spect=pred_spect,
            true_spect_len=spect_len,
            attn_logprob=attn_logprob,
            attn_soft=attn_soft,
            attn_hard=attn_hard,
            attn_hard_dur=attn_hard_dur,
        )

        train_log = {
            'train_loss': loss,
            'train_durs_loss': durs_loss,
            'train_pitch_loss': torch.tensor(1.0).to(durs_loss.device) if pitch_loss is None else pitch_loss,
            'train_mel_loss': mel_loss,
            'train_durs_acc': acc,
            'train_durs_acc_dist_3': acc_dist_3,
            'train_ctc_loss': torch.tensor(1.0).to(durs_loss.device) if ctc_loss is None else ctc_loss,
            'train_bin_loss': torch.tensor(1.0).to(durs_loss.device) if bin_loss is None else bin_loss,
        }

        return {'loss': loss, 'progress_bar': {k: v.detach() for k, v in train_log.items()}, 'log': train_log}

    def validation_step(self, batch, batch_idx, epoch):
        attn_prior = None
        spect, spect_len, text, text_len, pitch = batch

        # pitch normalization
        # zero_pitch_idx = pitch == 0
        # pitch = (pitch - self.pitch_mean) / self.pitch_std
        # pitch[zero_pitch_idx] = 0.0

        (pred_spect, _, pred_log_durs, pred_pitch, attn_soft, attn_logprob, attn_hard, attn_hard_dur,) = self(
            text=text,
            text_len=text_len,
            pitch=pitch,
            spect=spect,
            spect_len=spect_len,
            attn_prior=attn_prior
        )

        (loss, durs_loss, acc, acc_dist_1, acc_dist_3, pitch_loss, mel_loss, ctc_loss, bin_loss,) = self._metrics(
            pred_durs=pred_log_durs,
            pred_pitch=pred_pitch,
            true_durs=attn_hard_dur,
            true_text_len=text_len,
            true_pitch=pitch,
            true_spect=spect,
            pred_spect=pred_spect,
            true_spect_len=spect_len,
            attn_logprob=attn_logprob,
            attn_soft=attn_soft,
            attn_hard=attn_hard,
            attn_hard_dur=attn_hard_dur,
        )

        # without ground truth internal features except for durations
        pred_spect, _, pred_log_durs, pred_pitch, attn_soft, attn_logprob, attn_hard, attn_hard_dur = self(
            text=text,
            text_len=text_len,
            pitch=None,
            spect=spect,
            spect_len=spect_len,
            attn_prior=attn_prior
        )

        *_, with_pred_features_mel_loss, _, _ = self._metrics(
            pred_durs=pred_log_durs,
            pred_pitch=pred_pitch,
            true_durs=attn_hard_dur,
            true_text_len=text_len,
            true_pitch=pitch,
            true_spect=spect,
            pred_spect=pred_spect,
            true_spect_len=spect_len,
            attn_logprob=attn_logprob,
            attn_soft=attn_soft,
            attn_hard=attn_hard,
            attn_hard_dur=attn_hard_dur,
        )

        val_log = {
            'val_loss': loss,
            'val_durs_loss': durs_loss,
            'val_pitch_loss': torch.tensor(1.0).to(durs_loss.device) if pitch_loss is None else pitch_loss,
            'val_mel_loss': mel_loss,
            'val_with_pred_features_mel_loss': with_pred_features_mel_loss,
            'val_durs_acc': acc,
            'val_durs_acc_dist_3': acc_dist_3,
            'val_ctc_loss': torch.tensor(1.0).to(durs_loss.device) if ctc_loss is None else ctc_loss,
            'val_bin_loss': torch.tensor(1.0).to(durs_loss.device) if bin_loss is None else bin_loss,
        }

        ret = {}
        ret['log'] = val_log

        if batch_idx == 0:
            specs = []
            pitches = []
            audios = []
            for i in range(min(3, spect.shape[0])):
                specs += [
                    [
                        plot_spectrogram_to_numpy(spect[i, :, : spect_len[i]].data.cpu().numpy()),
                        f"gt mel {i}",
                    ],
                    [
                        plot_spectrogram_to_numpy(pred_spect.transpose(1, 2)[i, :, : spect_len[i]].data.cpu().numpy()),
                        f"pred mel {i}",
                    ],
                ]

                pitches += [
                    [
                        plot_pitch_to_numpy(
                            average_pitch(pitch.unsqueeze(1), attn_hard_dur)
                            .squeeze(1)[i, : text_len[i]]
                            .data.cpu()
                            .numpy(),
                            ylim_range=[-2.5, 2.5],
                        ),
                        f"gt pitch {i}",
                    ],
                ]

                pitches += [
                    [
                        plot_pitch_to_numpy(pred_pitch[i, : text_len[i]].data.cpu().numpy(), ylim_range=[-2.5, 2.5]),
                        f"pred pitch {i}",
                    ],
                ]

                if epoch > 10:
                    audios += [
                        [
                            np.expand_dims(griffin_lim(spect[i, :, : spect_len[i]].data.cpu().numpy()), axis=0),
                            f"gt audio {i}"
                        ],
                        [
                            np.expand_dims(griffin_lim(pred_spect.transpose(1, 2)[i, :, : spect_len[i]].data.cpu().numpy()), axis=0),
                            f"pred audio {i}"
                        ],
                    ]
                else:
                    audios += [None, None]
                

            ret['specs'] = specs
            ret['pitches'] = pitches
            ret['audios'] = audios

        return ret

    """
    def generate_spectrogram(
        self,
        tokens: Optional[torch.Tensor] = None,
        tokens_len: Optional[torch.Tensor] = None,
        lm_tokens: Optional[torch.Tensor] = None,
        raw_texts: Optional[List[str]] = None,
        norm_text_for_lm_model: bool = True,
        lm_model: str = "albert",
    ):
        if tokens is not None:
            if tokens_len is None:
                # It is assumed that padding is consecutive and only at the end
                tokens_len = (tokens != self.tokenizer.pad).sum(dim=-1)
        else:
            if raw_texts is None:
                raise ValueError("raw_texts must be specified if tokens is None")

            t_seqs = [self.tokenizer(t) for t in raw_texts]
            tokens = torch.nn.utils.rnn.pad_sequence(
                sequences=[torch.tensor(t, dtype=torch.long, device=self.device) for t in t_seqs],
                batch_first=True,
                padding_value=self.tokenizer.pad,
            )
            tokens_len = torch.tensor([len(t) for t in t_seqs], dtype=torch.long, device=tokens.device)

        if self.cond_on_lm_embeddings and lm_tokens is None:
            if raw_texts is None:
                raise ValueError("raw_texts must be specified if lm_tokens is None")

            lm_model_tokenizer = self._get_lm_model_tokenizer(lm_model)
            lm_padding_value = lm_model_tokenizer._convert_token_to_id('<pad>')
            lm_space_value = lm_model_tokenizer._convert_token_to_id('▁')

            assert isinstance(self.tokenizer, EnglishCharsTokenizer) or isinstance(
                self.tokenizer, EnglishPhonemesTokenizer
            )

            if norm_text_for_lm_model and self.text_normalizer_call is not None:
                raw_texts = [self.text_normalizer_call(t, **self.text_normalizer_call_kwargs) for t in raw_texts]

            preprocess_texts_as_tts_input = [self.tokenizer.text_preprocessing_func(t) for t in raw_texts]
            lm_tokens_as_ids_list = [
                lm_model_tokenizer.encode(t, add_special_tokens=False) for t in preprocess_texts_as_tts_input
            ]

            if self.tokenizer.pad_with_space:
                lm_tokens_as_ids_list = [[lm_space_value] + t + [lm_space_value] for t in lm_tokens_as_ids_list]

            lm_tokens = torch.full(
                (len(lm_tokens_as_ids_list), max([len(t) for t in lm_tokens_as_ids_list])),
                fill_value=lm_padding_value,
                device=tokens.device,
            )
            for i, lm_tokens_i in enumerate(lm_tokens_as_ids_list):
                lm_tokens[i, : len(lm_tokens_i)] = torch.tensor(lm_tokens_i, device=tokens.device)

        pred_spect = self.infer(tokens, tokens_len, lm_tokens=lm_tokens).transpose(1, 2)
        return pred_spect
        """
