import torch
import torch.nn as nn


def MLP(num_in, num_hidden, num_out):
    return nn.Sequential(
        nn.Linear(num_in, num_hidden),
        nn.GELU(),
        nn.Linear(num_hidden, num_out),
    )


class Position1d(nn.Module):
    def __init__(self, dims) -> None:
        super().__init__()
        scales = torch.pow(1000, -2 * torch.arange(dims // 2) / dims)
        weight = torch.repeat_interleave(scales, 2)[None, :]
        bias = torch.zeros_like(weight)
        bias[1::2] = torch.pi / 2
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)

    def forward(self, position):
        return torch.sin(position @ self.weight + self.bias)


class HyperTokenMixer(nn.Module):
    def __init__(self, d, d_prime=None, tied=True, hyper_net=None, act=None):
        super().__init__()
        if not d_prime:
            d_prime = d * 2
        if not hyper_net:
            hyper_net = lambda d, d_prime: MLP(d, d_prime, d_prime)

        self.h1 = hyper_net(d, d_prime)
        self.h2 = hyper_net(d, d_prime) if not tied else None
        self.activation = nn.GELU() if not act else act

    def forward(self, X, P):
        X.add_(P)

        W1 = self.h1(X)
        W2 = self.h2(X) if self.h2 else W1

        P = torch.transpose(W1, 1, 2) @ X
        A = self.activation(P)

        Y = W2 @ A
        return Y


class HyperMixerLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.token_mixer = HyperTokenMixer(d)
        self.feature_mixer = MLP(d, d, d)

    def forward(self, X, P=None):
        X = X + self.token_mixer(self.norm(X), P)
        X = X + self.feature_mixer(self.norm(X))
        return X


class HyperMixer(nn.Module):
    def __init__(self, *, d=256, layers=8, layer_fun=None, n_classes=1000):
        super().__init__()
        if not layer_fun:
            layer_fun = HyperMixerLayer

        self.position_encoder = Position1d(d)

        self.layers = nn.ModuleList([layer_fun(d) for _ in range(layers)])
        self.supervised = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, n_classes),
        )

    def forward(self, X, P):
        P = self.position_encoder(P)

        for layer in self.layers:
            X = layer(X, P)
        Y = self.supervised(torch.mean(X, dim=1))
        return Y
