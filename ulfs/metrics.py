import math
from typing import List, Iterable, Dict, Tuple, Hashable
from collections import defaultdict, Counter

import torch
import scipy.stats
import numpy as np


def lech_dist(A, B):
    """
    given two tensors A, and B, with the item index along the first dimension,
    and each tensor is 2-dimensional, this will calculate the lechenstein distance
    between each pair of examples between A and B

    both A and B are assumed to be long tensors of indices
    (cf one-hot)
    """
    assert A.dtype == torch.int64
    assert B.dtype == torch.int64

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


def lech_dist_from_samples(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """
    left and right are two sets of samples from the same
    tensor. they should both be two dimensional. dim 0
    is the sample index. dim 1 is the dimension we will
    calculate lechenstein distances over

    both left and right are assumed to be long tensors of indices
    (cf one-hot)
    """
    assert left.dtype == torch.int64
    assert right.dtype == torch.int64

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
        idxes = torch.ones(length * length, 2, dtype=torch.int64)
        idxes = idxes.cumsum(dim=0) - 1
        idxes[:, 0] = idxes[:, 0] // length
        idxes[:, 1] = idxes[:, 1] % length
    else:
        # we sample with replacement, assuming number of
        # possible pairs >> max_samples
        a_idxes = torch.from_numpy(np.random.choice(
            length, max_samples, replace=True))
        b_idxes = torch.from_numpy(np.random.choice(
            length, max_samples, replace=True))
        idxes = torch.stack([a_idxes, b_idxes], dim=1)
    return idxes


def topographic_similarity(utts: torch.Tensor, labels: torch.Tensor, max_samples=10000):
    """
    (quoting Angeliki 2018)
    "The intuition behind this measure is that semantically similar objects should have similar messages."

    a and b should be discrete; 2-dimensional. with item index along first dimension, and attribute index
    along second dimension

    if there are more pairs of utts and labels than max_samples, then sample pairs

    Parameters
    ----------
    utts: torch.Tensor
        assumed to be long tensor
    labels: torch.Tensor
        assumed to be long tensor
    """
    assert utts.size(0) == labels.size(0)
    assert len(utts.size()) == 2
    assert len(labels.size()) == 2
    assert utts.dtype == torch.int64
    assert labels.dtype == torch.int64

    pair_idxes = get_pair_idxes(utts.size(0), max_samples=max_samples)

    utts_left, utts_right = utts[pair_idxes[:, 0]], utts[pair_idxes[:, 1]]
    labels_left, labels_right = labels[pair_idxes[:, 0]], labels[pair_idxes[:, 1]]

    utts_pairwise_dist = lech_dist_from_samples(utts_left, utts_right)
    labels_pairwise_dist = lech_dist_from_samples(labels_left, labels_right)

    rho, _ = scipy.stats.spearmanr(a=utts_pairwise_dist.cpu(), b=labels_pairwise_dist.cpu())
    if rho != rho:
        # if rho is nan, we'll assume taht utts was all the same value. hence rho
        # is zero. (if labels was all the same value too, rho would be unclear, but
        # since the labels are provided by the dataset, we'll assume that they are diverse)
        max_utts_diff = (utts_pairwise_dist - utts_pairwise_dist[0]).abs().max().item()
        max_labels_diff = (labels_pairwise_dist - labels_pairwise_dist[0]).abs().max().item()
        print('rho is zero, max_utts_diff', max_utts_diff, 'max_labels_diff', max_labels_diff)
        rho = 0
    return rho.item()


def uniqueness(a: torch.Tensor) -> float:
    """
    given 2 dimensional discrete tensor a, will count the number of unique vectors,
    and divide by the total number of vectors, ie returns the fraction of vectors
    which are unique

    Parameters
    ----------
    a: torch.Tensor
        should be long tensor of indices
        should be dimensions [N][K]
    """
    assert a.dtype == torch.int64
    v = set()
    if len(a.size()) != 2:
        raise ValueError('size of a should be 2-dimensions, but a.size() is ' + str(a.size()))
    N, K = a.size()
    for n in range(N):
        v.add(','.join([str(x) for x in a[n].tolist()]))
    return (len(v) - 1) / (N - 1)   # subtract 1, because if everything is identical, there would still be 1


def cluster_strings(strings: Iterable[str]) -> torch.Tensor:
    """
    given a list of strings, assigns a clustering, where
    each pair of identical ground truth strings is in the same
    cluster
    return a torch.LongTensor containing the cluster id of
    each ground truth
    """
    cluster_id_by_truth: Dict[str, int] = {}
    cluster_l: List[int] = []
    for n, truth in enumerate(strings):
        cluster_id = cluster_id_by_truth.setdefault(truth, len(cluster_id_by_truth))
        cluster_l.append(cluster_id)
    return torch.tensor(cluster_l, dtype=torch.int64)


def cluster_utts(utts: torch.Tensor) -> torch.Tensor:
    """
    given a 2-d tensor of [S][N], where N is number of
    examples, and S is sequence length, and the tensor
    is of discrete int64 indices (cf distributions over
    tokens), we cluster all identical examples, and return
    a cluster assignment as a long tensor, containing the
    cluster id of each example, starting from 0

    if examples have differnet lengths, padding id should
    be identical. this function will compare the entire
    length of each example. as long as the padding id is
    consistent, this should work as desired, i.e. effectively
    ignore padding
    """
    S, N = utts.size()
    clustering = torch.zeros(N, dtype=torch.int64)
    seen = torch.zeros(N, dtype=torch.bool)
    cluster_id = 0
    for n in range(N):
        if seen[n]:
            continue
        mask = (utts == utts[:, n:n + 1]).all(dim=0)
        clustering[mask] = cluster_id
        cluster_id += 1
        seen[mask] = True
    return clustering


def calc_cluster_prec_recall(pred: torch.Tensor, ground: torch.Tensor) -> Tuple[float, float]:
    """
    given predicted clustering, and ground clustering,
    calculates cluster recall and precision
    """
    assert len(pred.size()) == 1
    assert len(ground.size()) == 1

    N = ground.size(0)
    assert pred.size(0) == N

    left_indices = torch.ones(N, dtype=torch.int64).cumsum(dim=0) - 1
    right_indices = torch.ones(N, dtype=torch.int64).cumsum(dim=0) - 1

    left_indices = left_indices.unsqueeze(-1).expand(N, N)
    right_indices = right_indices.unsqueeze(0).expand(N, N)

    left_indices = left_indices.contiguous().view(-1)
    right_indices = right_indices.contiguous().view(-1)

    dup_mask = left_indices <= right_indices

    ground_pos = ground[left_indices] == ground[right_indices]
    pred_pos = pred[left_indices] == pred[right_indices]

    tp = ((pred_pos & ground_pos) & dup_mask).sum().item()
    fp = ((pred_pos & (~ground_pos)) & dup_mask).sum().item()
    fn = (((~pred_pos) & ground_pos) & dup_mask).sum().item()
    tn = (((~pred_pos) & (~ground_pos)) & dup_mask).sum().item()

    assert tp + fp + fn + tn == N * (N + 1) / 2

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return precision, recall


def entropy(X: Iterable[Hashable]) -> float:
    """
    X should be list of hashable items
    """
    assert isinstance(X, list)
    freq_by_x: Dict[int, int] = defaultdict(int)
    N = len(X)
    for x in X:        # print('x', x, type(x))
        freq_by_x[x] += 1
    probs = torch.Tensor([count / N for count in freq_by_x.values()])
    H = - ((probs * probs.log()).sum() / torch.tensor([2.0]).log()).item()
    return H


def mutual_information(X: Iterable[Hashable], Y: Iterable[Hashable]) -> float:
    assert isinstance(X, list)
    assert isinstance(Y, list)
    N = len(X)
    assert len(Y) == N
    X_freq: Dict[Hashable, int] = defaultdict(int)
    Y_freq: Dict[Hashable, int] = defaultdict(int)
    XY_freq: Dict[Tuple[Hashable, Hashable], int] = defaultdict(int)
    for i in range(N):
        x = X[i]
        y = Y[i]
        X_freq[x] += 1
        Y_freq[y] += 1
        XY_freq[(x, y)] += 1
    X_probs = {x: freq / N for x, freq in X_freq.items()}
    Y_probs = {y: freq / N for y, freq in Y_freq.items()}
    XY_probs = {xy: freq / N for xy, freq in XY_freq.items()}
    I_sum = 0.0
    for x, y in XY_freq:
        xy = (x, y)
        I_sum += XY_probs[xy] * math.log(XY_probs[xy] / X_probs[x] / Y_probs[y])
    return I_sum / math.log(2)


def conditional_entropy(X: Iterable[Hashable], Y: Iterable[Hashable]) -> float:
    """
    returns H(X | Y)

    inputs are lists, of hashable items
    """
    assert isinstance(X, list)
    assert isinstance(Y, list)
    N = len(X)
    assert len(Y) == N
    X_set = set()
    Y_freq: Dict[Hashable, int] = defaultdict(int)
    XY_freq: Dict[Tuple[Hashable, Hashable], int] = defaultdict(int)
    for i in range(N):
        x = X[i]
        y = Y[i]
        X_set.add(x)
        Y_freq[y] += 1
        XY_freq[(x, y)] += 1
    Y_probs = {y: freq / N for y, freq in Y_freq.items()}
    XY_probs = {xy: freq / N for xy, freq in XY_freq.items()}
    H_sum = 0.0
    for x, y in XY_freq:
        xy = (x, y)
        H_sum += XY_probs[xy] * math.log(XY_probs[xy] / Y_probs[y])
    return - H_sum / math.log(2)


def to_base(value: int, base: int, length: int):
    repr = np.base_repr(value, base=base).zfill(length)
    repr = [ord(v) - ord('0') for v in repr]
    return repr


def generate_partitions(length: int, num_partitions: int):
    if num_partitions == 1:
        yield [0]
        return
    N = int(math.pow(num_partitions, length))
    for n in range(N):
        repr = to_base(n, num_partitions, length)
        # skip = False
        # for i in range(num_partitions):
        # if i not in repr:
        # skip = True
        # if skip:
        # continue
        yield repr


def pos_dis(meanings: torch.Tensor, utts: torch.Tensor):
    """
    posdis, Chaabouni et al 2020
    """
    N, num_atts = meanings.size()
    _N, msg_len = utts.size()
    assert N == _N
    # I is mutual information between each attribute
    # and each symbol in message
    I = torch.zeros(num_atts, msg_len)  # noqa: E741
    for att_idx in range(num_atts):
        for pos in range(msg_len):
            I[att_idx, pos] = mutual_information(
                meanings[:, att_idx].tolist(), utts[:, pos].tolist())

    posdis = 0.0
    for pos in range(msg_len):
        # mutual information between symbol at pos and the
        # top two attributes with the biggest mutual information
        i_01, a_01 = I[:, pos].topk(dim=0, k=2)
        # normalizing term
        H_pos = entropy(utts[:, pos].tolist())
        if H_pos != 0:
            posdis += (i_01[0] - i_01[1]).item() / H_pos / msg_len
    return posdis


def bos_dis(meanings: torch.Tensor, utts: torch.Tensor, vocab_size: int):
    """
    bosdis, Chaabouni et al 2020
    """
    N, num_atts = meanings.size()
    _N, _ = utts.size()
    _vocab_size = utts.max().item() + 1
    assert _vocab_size <= vocab_size
    assert N == _N

    symbol_counts = torch.zeros(N, vocab_size, dtype=torch.int64)
    for n in range(N):
        counts = Counter(utts[n].tolist())
        for sym, count in counts.items():
            symbol_counts[n, sym] = count
    utts = None  # prevent us using it by accident

    I = torch.zeros(num_atts, vocab_size)  # noqa: E741
    for att_idx in range(num_atts):
        for sym_idx in range(vocab_size):
            I[att_idx, sym_idx] = mutual_information(
                meanings[:, att_idx].tolist(), symbol_counts[:, sym_idx].tolist())

    bosdis = 0.0
    for sym_idx in range(vocab_size):
        # mutual information between symbol sym_idx and the
        # top two attributes with the biggest mutual information
        i_01, a_01 = I[:, sym_idx].topk(dim=0, k=2)
        # normalizing term
        H_sym = entropy(symbol_counts[:, sym_idx].tolist())
        if H_sym != 0:
            bosdis += (i_01[0] - i_01[1]).item() / H_sym / vocab_size
    return bosdis


def res_ent(meanings: torch.Tensor, utts: torch.Tensor, normalize: bool = True):
    """
    residual entropy, Resnick et al 2020
    """
    N, Na = meanings.size()
    _N, S = utts.size()
    assert N == _N
    assert S >= Na
    re = None
    best_p = None
    for p in generate_partitions(length=S, num_partitions=Na):
        _re_sum = 0.0
        for i in range(Na):
            idxes = [j for j, v in enumerate(p) if v == i]
            ci = meanings[:, i].tolist()
            zp = [tuple(v) for v in utts[:, idxes].tolist()]
            H_ci = entropy(ci)
            H_ci_zp = conditional_entropy(ci, zp)
            if H_ci != 0 and normalize:
                _re_bit = H_ci_zp / H_ci
                _re_sum += _re_bit
        _re = _re_sum / Na
        if re is None or _re < re:
            best_p = p
            re = _re
    assert re is not None
    return re, best_p


def res_ent_greedy(meanings: torch.Tensor, utts: torch.Tensor, normalize: bool = True):
    """
    residual entropy, Resnick et al 2020

    we calculate mutual information between each meaning dimension and each
    utterance dimension, and partition utterance dimension by taking max over
    meaning dimension over mutual information
    """
    N, Na = meanings.size()
    _N, S = utts.size()
    assert N == _N
    assert S >= Na
    re = None

    I = torch.zeros(Na, S)  # noqa: E741
    for i in range(Na):
        _meaning = meanings[:, i]
        for j in range(S):
            _I = mutual_information(_meaning.tolist(), utts[:, j].tolist())
            I[i, j] = _I
    _, rank = I.max(dim=0)
    print(rank)

    p = rank

    _re_sum = 0.0
    for i in range(Na):
        idxes = [j for j, v in enumerate(p) if v == i]
        ci = meanings[:, i].tolist()
        zp = [tuple(v) for v in utts[:, idxes].tolist()]
        H_ci = entropy(ci)
        H_ci_zp = conditional_entropy(ci, zp)
        if H_ci != 0:
            _re_bit = H_ci_zp
            if normalize:
                _re_bit = _re_bit / H_ci
            _re_sum += _re_bit
    re = _re_sum / Na
    return re, p


def compositional_entropy(meanings: torch.Tensor, utts: torch.Tensor, normalize: bool = True) -> float:
    return 1 - res_ent_greedy(meanings=meanings, utts=utts, normalize=normalize)[0]


def get_residual_entropy_orig(lang, target, num_att_vals: int):
    """
    This is forked from the code at
    https://github.com/backpropper/cbc-emecom/blob/6d01f0cdda4a8f742232b537da4f2633613f44a9/utils.py#L164-L204
    which was provided under an MIT license

    Modified:
    - returns log2 entropy
    - num_att_vals is parameterizable
    """
    num_bits = lang.shape[1]
    num_digits = target.shape[1]

    speaker = np.zeros((num_att_vals * num_digits, lang.shape[1] * 2))
    for (t, l) in zip(target, lang):
        for num, bit in enumerate(l):
            for char in t:
                speaker[char][num * 2 + int(bit)] += 1
    spknorm = np.zeros((num_att_vals * num_digits, lang.shape[1] * 2))
    for char in range(num_att_vals * num_digits):
        for i in range(0, lang.shape[1] * 2, 2):
            spknorm[char][i] = speaker[char][i] / (speaker[char][i] + speaker[char][i + 1])
            spknorm[char][i + 1] = speaker[char][i + 1] / (speaker[char][i] + speaker[char][i + 1])

    spkprobs_0 = spknorm[:, ::2]
    spkprobs_0_diff = abs(spkprobs_0 - spkprobs_0.mean(axis=0))
    ranks = np.zeros((num_digits, num_bits))
    for category in range(num_digits):
        ranks[category] = np.mean(spkprobs_0_diff[category * num_att_vals:(category + 1) * num_att_vals], axis=0)
    indx: List[List[int]] = [[] for _ in range(num_digits)]
    for b in range(num_bits):
        v = np.argmax(ranks[:, b])
        indx[int(v)].append(b)
    ents = np.zeros(num_digits)
    for cat in range(num_digits):
        dat = lang[:, indx[cat]]
        indx_length = len(indx[cat])
        vec = 2 ** np.array(range(indx_length))
        probs_ = np.zeros((2 ** indx_length, num_att_vals)) + 1e-8
        probs_n_ = np.zeros((2 ** indx_length, num_att_vals)) + 1e-8
        fp = dat.dot(vec)  # the number formed if the language is binary input, eg [1,0,1] becomes 5
        for i in range(len(fp)):
            probs_[int(fp[i]), target[i, cat] - cat * num_att_vals] += 1
        probs_n_ = probs_ / (np.reshape(np.sum(probs_, axis=1), [-1, 1]))
        ent = np.sum(probs_n_ * np.log(probs_n_) / np.log(2), 1)
        ents[cat] = np.sum(ent * (probs_.sum(1) / probs_.sum()))
    ent = -np.mean(ents)
    ent = ent.item()
    return ent
