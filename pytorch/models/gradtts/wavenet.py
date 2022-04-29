import torch
import torch.nn as nn
import torch.nn.functional as F


class StepEncoder(nn.Module):
    def __init__(self, dim, max_steps=100):
        super().__init__()
        self.fc1 = nn.Linear(128, dim)
        self.fc2 = nn.Linear(dim, dim)
        steps = torch.arange(max_steps).unsqueeze(1)
        dims = torch.arange(64).unsqueeze(0)
        table = steps * 10.0**(dims * 4.0 / 63.0)
        self.register_buffer('encoding', torch.cat([torch.sin(table), torch.cos(table)], dim=1), persistent=False)

    def forward(self, t):
        x = self.encoding[t]
        x = self.fc1(x)
        x = F.silu(x)
        x = self.fc2(x)
        x = F.silu(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, enc_dim, dim):
        super().__init__()
        self.conditioner = nn.Conv1d(enc_dim, dim * 2, kernel_size=1)
        self.gau = nn.Conv1d(dim, dim * 2, kernel_size=3, padding=1)
        self.main = nn.Conv1d(dim, dim, kernel_size=1)
        self.skip = nn.Conv1d(dim, dim, kernel_size=1)
        self.scale = 2.0 ** 0.5

    def forward(self, x, e, t, mask):
        o = x + t
        o = self.gau(o * mask)
        o = o + self.conditioner(e)
        a, b = torch.chunk(o, 2, dim=1)
        o = torch.tanh(a) * torch.sigmoid(b) * mask
        skip = self.skip(o)
        o = self.main(o) * mask
        o = o + x
        return o / self.scale, skip


class Wavenet(nn.Module):
    def __init__(self, n_mels=80, enc_dim=192, n_layers=20, dim=256, max_steps=100):
        super().__init__()
        self.step_encoder = StepEncoder(dim, max_steps)
        self.residual_blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.residual_blocks.append(ResidualBlock(enc_dim, dim))
        self.pre_conv = nn.Conv1d(n_mels, dim, kernel_size=1)
        self.skip_conv1 = nn.Conv1d(dim, dim, kernel_size=1)
        self.skip_conv2 = nn.Conv1d(dim, n_mels, kernel_size=1)
        self.scale = n_layers ** 0.5

    def forward(self, x, mask, e, t):
        e = e * mask
        t = self.step_encoder(t).unsqueeze(-1)
        x = self.pre_conv(x * mask) * mask
        x = F.relu(x)
        skips = 0
        for b in self.residual_blocks:
            x, skip = b(x, e, t, mask)
            skips = skips + skip
        x = self.skip_conv1(skips * mask / self.scale)
        x = F.relu(x)
        x = self.skip_conv2(x * mask)
        return x * mask


"""
b1 = 10-4
bt = 0.02
bmin = 0.1
bmax = 20

bi = bmin/N + (i-1)/(N(N-1))(bmax-bmin)

0.1 / 1000 = 0.
0.1 /1000 + (1000 - 1)/(1000*999)(20-0.1)
"""


class Decoder(nn.Module):
    def __init__(self, n_mels, enc_dim, n_layers, dim, max_steps, loss_fn):
        super().__init__()
        self.decoder = Wavenet(n_mels, enc_dim, n_layers, dim, max_steps)
        beta = torch.linspace(1e-4, 0.06, steps=max_steps)
        alpha = 1 - beta
        alpha_hat = torch.cumprod(alpha, dim=0)
        self.register_buffer('beta', beta, persistent=False)
        self.register_buffer('alpha', alpha, persistent=False)
        self.register_buffer('alpha_hat', alpha_hat, persistent=False)
        self.loss_fn = loss_fn

    def forward_diffusion(self, x0, mask, mu, t):
        #x0: mel, mask: mask, mu: encoder, t: t
        alpha_hat_t = self.alpha_hat[t].view(-1, 1, 1)
        epsilon = torch.randn_like(x0)
        c = (1 - alpha_hat_t)**0.5 * epsilon
        xt = alpha_hat_t**0.5 * x0 + c
        return xt * mask, epsilon * mask

    @torch.no_grad()
    def reverse_diffusion(self, z, mask, mu, n_timesteps, stoc=False, spk=None):
        xt = z * mask
        for t in reversed(range(len(self.alpha_hat))):
            tt = torch.tensor([t]*z.shape[0])
            e_theta = self.decoder(xt, mask, mu, tt)
            alpha_t = self.alpha[tt].view(-1, 1, 1)
            alpha_hat_t = self.alpha_hat[tt].view(-1, 1, 1)
            c1 = 1 / alpha_t**0.5
            c2 = (1 - alpha_t) / (1 - alpha_hat_t)**0.5
            xt = c1 * (xt - (c2 * e_theta))
            if t > 0:
                beta_t = self.beta[tt].view(-1, 1, 1)
                z = torch.randn_like(xt)
                alpha_hat_t_1 = self.alpha_hat[tt-1].view(-1, 1, 1)
                sigma_t = ((1 - alpha_hat_t_1) / (1 - alpha_hat_t) * beta_t)**0.5
                xt = xt + sigma_t * z
            xt = xt * mask
        return xt

    @torch.no_grad()
    def forward(self, z, mask, mu, n_timesteps, stoc=False, spk=None):
        return self.reverse_diffusion(z, mask, mu, n_timesteps, stoc, spk)

    def loss_t(self, x0, mask, mu, t, spk=None):
        xt, z = self.forward_diffusion(x0, mask, mu, t)
        e_theta = self.decoder(xt, mask, mu, t)
        loss = self.loss_fn(z, e_theta)
        return loss, xt

    def compute_loss(self, x0, mask, mu, spk=None, offset=1e-5):
        t = torch.randint(0, len(self.beta), [x0.shape[0]])
        return self.loss_t(x0, mask, mu, t, spk)
