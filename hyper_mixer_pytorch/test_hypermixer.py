import pytest
import torch
from .hypermixer import HyperMixerLayer


def test_tied_parameters():
    d, d_prime = 256, 512

    tied = HyperMixerLayer(d, d_prime, tied=True)
    untied = HyperMixerLayer(d, d_prime, tied=False)

    assert sum(p.numel() for p in tied.parameters()) < sum(
        p.numel() for p in untied.parameters()
    )


def test_token_permutation():
    N, d, d_prime = 100, 32, 64

    T = torch.randn(1, N, d)
    P = torch.randn(1, N, d)
    perm = torch.randperm(N)
    layer = HyperMixerLayer(d, d_prime, tied=True)

    Y1 = layer(T, P)[:, perm, :]
    Y2 = layer(T[:, perm, :], P[:, perm, :])

    assert torch.allclose(Y1, Y2, atol=1e-5)


@pytest.mark.skip(reason="TODO")
def test_full_model():
    raise NotImplementedError()


@pytest.mark.skip(reason="TODO")
def test_position_encoding():
    raise NotImplementedError()
