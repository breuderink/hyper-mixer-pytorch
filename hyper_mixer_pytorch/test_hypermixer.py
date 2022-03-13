from .hypermixer import HyperMixerLayer
import torch


def test_tied_parameters():
    d, d_prime = 256, 512

    tied = HyperMixerLayer(d, d_prime, tied=True)
    untied = HyperMixerLayer(d, d_prime, tied=False)

    assert sum(p.numel() for p in tied.parameters()) < sum(
        p.numel() for p in untied.parameters()
    )


def test_token_permutation():
    N, d, d_prime = 100, 32, 64

    X = torch.randn(1, N, d)
    perm = torch.randperm(N)
    layer = HyperMixerLayer(d, d_prime, tied=True)

    Y1 = layer(X)[:,perm]
    Y2 = layer(X[:,perm])

    assert torch.allclose(Y1, Y2, atol=1e-5)
