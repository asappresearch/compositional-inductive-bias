import torch
import numpy as np

from ulfs import metrics


def test_topographic_similarity():
    N = 32
    K = 5
    M = 6
    a = torch.from_numpy(np.random.choice(K, (N, M), replace=True))
    b = torch.from_numpy(np.random.choice(K, (N, M), replace=True))
    rho = metrics.topographic_similarity(a, b)
    print('rho %.3f' % rho)

    rho = metrics.topographic_similarity(a, a)
    print('rho %.3f' % rho)

    rho = metrics.topographic_similarity(b, b)
    print('rho %.3f' % rho)

    rho = metrics.topographic_similarity(a, torch.cat([a[:N // 2], b[N // 2:]]))
    print('rho %.3f' % rho)

    rho = metrics.topographic_similarity(a, a // 2)
    print('rho %.3f' % rho)

    rho = metrics.topographic_similarity(a, a // 4)
    print('rho %.3f' % rho)

    rho = metrics.topographic_similarity(a, torch.zeros(N, M).long())
    print('rho %.3f' % rho)


def test_uniqueness():
    N = 50
    K = 4
    V = 100
    a = torch.from_numpy(np.random.choice(V, (N, K), replace=True))
    print('uniqueness', metrics.uniqueness(a))
    a[:N // 5] = a[0]
    print('uniqueness', metrics.uniqueness(a))
    a[:] = a[0]
    print('uniqueness', metrics.uniqueness(a))
