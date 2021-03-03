import torch
import scipy.stats
import numpy as np


def lech_dist(A, B):
    """
    given two tensors A, and B, with the item index along the first dimension,
    and each tensor is 2-dimensional, this will calculate the lechenstein distance
    between each pair of examples between A and B
    """
    N_a = A.size(0)
    N_b = B.size(0)
    E = A.size(1)
    assert E == B.size(1)
    assert len(A.size()) == 2
    assert len(B.size()) == 2
    if N_a * N_b * 4 / 1000 / 1000 >= 500:  # if use > 500MB memory, then die
        raise Exception('Would use too much memory => dieing')
    A = A.unsqueeze(1).expand(N_a, N_b, E)
    B = B.unsqueeze(0).expand(N_a, N_b, E)
    AeqB = A == B
    dists = AeqB.sum(dim=-1)
    dists = dists.float() / E

    return dists


def lech_dist_from_samples(left, right):
    """
    left and right are two sets of samples from the same
    tensor. they should both be two dimensional. dim 0
    is the sample index. dim 1 is the dimension we will
    calculate lechenstein distances over
    """
    assert len(left.size()) == 2
    assert len(right.size()) == 2

    N = left.size(0)
    assert right.size(0) == N
    E = left.size(1)
    assert E == right.size(1)

    left_eq_right = left == right
    dists = left_eq_right.sum(dim=-1)
    dists = dists.float() / E

    return dists


def tri_to_vec(tri):
    """
    returns lower triangle of a square matrix, as a vector, excluding the diagonal

    eg given

    1 3 9
    4 3 7
    2 1 5

    returns:

    4 2 1
    """
    assert len(tri.size()) == 2
    assert tri.size(0) == tri.size(1)
    K = tri.size(0)
    res_size = (K - 1) * K // 2
    res = torch.zeros(res_size, dtype=tri.dtype)
    pos = 0
    for k in range(K - 1):
        res[pos:pos + (K - k - 1)] = tri[k + 1:, k]
        pos += (K - k - 1)
    return res


def calc_squared_euc_dist(one, two):
    """
    input: two arrays, [N1][E]
                       [N2][E]
    output: one matrix: [N1][N2]
    """
    one_squared = (one * one).sum(dim=1)
    two_squared = (two * two).sum(dim=1)
    transpose = one @ two.transpose(0, 1)
    squared_dist = one_squared.unsqueeze(1) + two_squared.unsqueeze(0) - 2 * transpose
    return squared_dist


def get_pair_idxes(length, max_samples):
    """
    return pairs of indices
    each pair should be unique
    no more than max_samples pairs will be returned
    (will sample if length * length > max_samples)
    """
    if length * length <= max_samples:
        # print('exhaustive all pairs')
        idxes = torch.ones(length * length, 2, dtype=torch.int64)
        idxes = idxes.cumsum(dim=0) - 1
        idxes[:, 0] = idxes[:, 0] // length
        idxes[:, 1] = idxes[:, 1] % length
    else:
        # print('sampling idx pairs')
        # we sample with replacement, assuming number of
        # possible pairs >> max_samples
        a_idxes = torch.from_numpy(np.random.choice(
            length, max_samples, replace=True))
        b_idxes = torch.from_numpy(np.random.choice(
            length, max_samples, replace=True))
        idxes = torch.stack([a_idxes, b_idxes], dim=1)
    return idxes


def topographic_similarity(utts, labels, max_samples=10000):
    """
    (quoting Angeliki 2018)
    "The intuition behind this measure is that semantically similar objects should have similar messages."

    a and b should be discrete; 2-dimensional. with item index along first dimension, and attribute index
    along second dimension

    if there are more pairs of utts and labels than max_samples, then sample pairs
    """
    assert utts.size(0) == labels.size(0)
    assert len(utts.size()) == 2
    assert len(labels.size()) == 2

    pair_idxes = get_pair_idxes(utts.size(0), max_samples=max_samples)

    utts_left, utts_right = utts[pair_idxes[:, 0]], utts[pair_idxes[:, 1]]
    labels_left, labels_right = labels[pair_idxes[:, 0]], labels[pair_idxes[:, 1]]

    utts_pairwise_dist = lech_dist_from_samples(utts_left, utts_right)
    labels_pairwise_dist = lech_dist_from_samples(labels_left, labels_right)

    rho, _ = scipy.stats.spearmanr(a=utts_pairwise_dist, b=labels_pairwise_dist)
    if rho != rho:
        # if rho is nan, we'll assume taht utts was all the same value. hence rho
        # is zero. (if labels was all the same value too, rho would be unclear, but
        # since the labels are provided by the dataset, we'll assume that they are diverse)
        max_utts_diff = (utts_pairwise_dist - utts_pairwise_dist[0]).abs().max().item()
        max_labels_diff = (labels_pairwise_dist - labels_pairwise_dist[0]).abs().max().item()
        print('rho is zero, max_utts_diff', max_utts_diff, 'max_labels_diff', max_labels_diff)
        rho = 0
    return rho


def uniqueness(a):
    """
    given 2 dimensional discrete tensor a, will count the number of unique vectors,
    and divide by the total number of vectors, ie returns the fraction of vectors
    which are unique
    """
    v = set()
    N, K = a.size()
    for n in range(N):
        v.add(','.join([str(x) for x in a[n].tolist()]))
    return (len(v) - 1) / (N - 1)   # subtract 1, because if everything is identical, there would still be 1
