import torch
import torch.nn as nn
import torch.nn.functional as F


def MLP(num_in, num_hidden=None, num_out=None):
    num_hidden = num_hidden if num_hidden else num_in
    num_out = num_out if num_out else num_in
    return nn.Sequential(
        nn.Linear(num_in, num_hidden),
        nn.GELU(),
        nn.Linear(num_hidden, num_out),
    )


class TokenMixer(nn.Module):
    def __init__(self, d, d_prime, tied):
        super().__init__()
        self.MLP1 = MLP(num_in=d, num_out=d_prime)
        self.MLP2 = MLP(num_in=d, num_out=d_prime) if not tied else None

    def forward(self, T, P):
        X = T + P
        W1 = self.MLP1(X)
        W2 = self.MLP2(X) if self.MLP2 else W1
        P = torch.einsum("bnp, bnd -> bpd", W1, X)
        A = F.gelu(P)
        Y = torch.einsum("bnp, bpd -> bnd", W2, A)
        return Y


class HyperMixerLayer(nn.Module):
    def __init__(self, d, d_prime, *, tied=True):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.token_mixer = TokenMixer(d=d, d_prime=d_prime, tied=tied)
        self.feature_mixer = MLP(d, d, d)

    def forward(self, T, P):
        T = T + self.token_mixer(self.norm(T), P)
        T = T + self.feature_mixer(self.norm(T))
        return T


class HyperMixer(nn.Module):
    def __init__(self, *, layers=8, d=256, d_prime=512, tied=True, n_classes=1000):
        super().__init__()
        self.layers = [HyperMixerLayer(d, d_prime, tied=tied) for _ in range(layers)]
        self.supervised = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, n_classes),
        )

    def forward(self, T, P):
        for layer in self.layers:
            T = layer(T, P)
        Y = self.supervised(torch.mean(T, dim=1))
        return Y
