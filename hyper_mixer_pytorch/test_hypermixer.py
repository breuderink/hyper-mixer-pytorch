import pytest
import torch
from .hypermixer import hyper_mixer_layer


def test_tied_parameters():
    d = 256

    tied = hyper_mixer_layer(d)
    untied = hyper_mixer_layer(d, tied=False)

    tied_params = sum(p.numel() for p in tied.parameters())
    untied_params = sum(p.numel() for p in untied.parameters())

    assert 1.5 * tied_params < untied_params


def test_token_permutation():
    N, d = 100, 32

    T = torch.randn(1, N, d)
    P = torch.randn(1, N, d)
    perm = torch.randperm(N)
    layer = hyper_mixer_layer(d)

    Y1 = layer(T, P)[:, perm, :]
    Y2 = layer(T[:, perm, :], P[:, perm, :])

    assert torch.allclose(Y1, Y2, atol=1e-5)
