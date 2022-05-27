import torch
import torch.nn as nn
from math import pi


def MLP(num_in, num_hidden, num_out):
    return nn.Sequential(
        nn.Linear(num_in, num_hidden),
        nn.GELU(),
        nn.Linear(num_hidden, num_out),
    )


class HyperTokenMixer(nn.Module):
    def __init__(self, *, d, d_prime, h1, h2=None, activation=None):
        super().__init__()
        self.h1 = h1
        self.h2 = h2
        self.activation = activation if activation else nn.GELU()

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
    def __init__(self, d, *, token_mixer, feature_mixer):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.token_mixer = token_mixer
        self.feature_mixer = feature_mixer

    def forward(self, X, P=None):
        X = X + self.token_mixer(self.norm(X), P)
        X = X + self.feature_mixer(self.norm(X))
        return X


def hyper_mixer_layer(d, *, d_prime=None, tied=True):
    d_prime = d_prime if d_prime else d * 2
    h1 = MLP(d, d, d_prime)
    h2 = None if tied else MLP(d, d, d_prime)
    return HyperMixerLayer(
        d,
        token_mixer=HyperTokenMixer(d=d, d_prime=d_prime, h1=h1, h2=h2),
        feature_mixer=MLP(d, d, d),
    )

class HyperMixer(nn.Module):
    def __init__(
        self, *, d=256, layers=8, layer=hyper_mixer_layer, n_classes=1000
    ):
        super().__init__()
        self.layers = nn.ModuleList([layer(d) for _ in range(layers)])
        self.supervised = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, n_classes),
        )

    def forward(self, X, P):
        for layer in self.layers:
            X = layer(X, P)

        Y = self.supervised(torch.mean(X, dim=1))
        return Y
