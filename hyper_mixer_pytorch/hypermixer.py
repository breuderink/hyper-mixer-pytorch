import torch
import torch.nn as nn
from math import pi


def MLP(num_in, num_hidden, num_out):
    return nn.Sequential(
        nn.Linear(num_in, num_hidden),
        nn.GELU(),
        nn.Linear(num_hidden, num_out),
    )


class Position1D(nn.Module):
    def __init__(self, dims) -> None:
        super().__init__()
        scales = torch.pow(1000, -2 * torch.arange(dims // 2) / dims)
        weight = torch.repeat_interleave(scales, 2)[None, :]
        bias = torch.zeros_like(weight)
        bias[:,1::2] = pi / 2
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)

    def forward(self, position):
        return torch.sin(position @ self.weight + self.bias)


class HyperTokenMixer(nn.Module):
    def __init__(self, d, *, d_prime=None, tied=True, hyper_net=None, activation=None):
        super().__init__()
        if not d_prime:
            d_prime = d * 2
        if not hyper_net:
            hyper_net = lambda d, d_prime: MLP(d, d_prime, d_prime)

        self.h1 = hyper_net(d, d_prime)
        self.h2 = hyper_net(d, d_prime) if not tied else None
        self.activation = nn.GELU() if not activation else activation

    def forward(self, X, P):
        # FIXME: the authors multiply X by sqrt(d) in code.
        X = X + P

        W1 = self.h1(X)
        W2 = self.h2(X) if self.h2 else W1

        P = torch.transpose(W1, 1, 2) @ X
        A = self.activation(P)

        Y = W2 @ A
        return Y


class HyperMixerLayer(nn.Module):
    def __init__(self, d, *, token_mixer=None, feature_mixer=None):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.token_mixer = token_mixer(d) if token_mixer else HyperTokenMixer(d)
        self.feature_mixer = feature_mixer(d) if feature_mixer else MLP(d, d, d)

    def forward(self, X, P=None):
        X = X + self.token_mixer(self.norm(X), P)
        X = X + self.feature_mixer(self.norm(X))
        return X


class HyperMixer(nn.Module):
    def __init__(
        self,
        *,
        position_encoder=Position1D,
        d=256,
        layers=8,
        layer=HyperMixerLayer,
        n_classes=1000
    ):
        super().__init__()
        self.position_encoder = position_encoder(d)
        self.layers = nn.ModuleList([layer(d) for _ in range(layers)])
        self.supervised = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, n_classes),
        )

    def forward(self, X, P=None):
        if P == None:
            P = torch.arange(X.shape[-2], dtype=X.dtype)[None, :, None]
        PE = self.position_encoder(P)

        for layer in self.layers:
            X = layer(X, PE)

        Y = self.supervised(torch.mean(X, dim=1))
        return Y
