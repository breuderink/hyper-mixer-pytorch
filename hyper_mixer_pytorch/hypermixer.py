from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F


def MLP(num_in, num_out, *, num_hidden=None):
    return nn.Sequential(
        nn.Linear(num_in, num_hidden),
        nn.GELU(),
        nn.Linear(num_hidden, num_out),
    )


class HyperTokenMixer(nn.Module):
    def __init__(self, d, d_prime, tied, hyper_net, act=None):
        super().__init__()
        self.h1 = hyper_net(d, d_prime)
        self.h2 = hyper_net(d, d_prime) if not tied else None
        self.activation = nn.GELU() if not act else act

    def forward(self, X, P):
        X = X + P
        W1 = self.h1(X)
        W2 = self.h2(X) if self.h2 else W1
        P = torch.einsum("bnp, bnd -> bpd", W1, X)
        A = self.activation(P)
        Y = torch.einsum("bnp, bpd -> bnd", W2, A)
        return Y


class HyperMixerLayer(nn.Module):
    def __init__(self, d, d_prime, *, f=None, tied=None):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        if not f:
            f = partial(MLP, num_hidden=d_prime)
        self.token_mixer = HyperTokenMixer(d, d_prime, tied, f)
        self.feature_mixer = f(d, d)

    def forward(self, X, P):
        X = X + self.token_mixer(self.norm(X), P)
        X = X + self.feature_mixer(self.norm(X))
        return X


class HyperMixer(nn.Module):
    def __init__(
        self,
        *,
        embedding=None,
        layers=8,
        d=256,
        d_prime=512,
        tied=True,
        f=None,
        n_classes=1000
    ):
        super().__init__()
        self.embedding = embedding
        self.layers = [
            HyperMixerLayer(d, d_prime, f=f, tied=tied) for _ in range(layers)
        ]
        self.supervised = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, n_classes),
        )

    def forward(self, X, P):
        if self.embedding:
            X = self.embedding(X)
        for layer in self.layers:
            X = layer(X, P)
        Y = self.supervised(torch.mean(X, dim=1))
        return Y


def position_1d(position, dims):
    """Position has shape (b, t, 1) or (t, 1)."""
    scales = torch.pow(1000, -2 * torch.arange(dims // 2) / dims)
    assert scales.numel() * 2 == dims
    S = position * scales
    return torch.cat((torch.sin(S), torch.cos(S)), dim=-1)
