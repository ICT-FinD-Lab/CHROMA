"""Exact float32 all-pairs AUROC without storing a quadratic score matrix.

For each score x, w(x)=2*#positive_scores>x + #positive_scores==x.
Summing w over all N*N scores includes the positive-positive contribution P*P.
Thus AUROC=(sum_all(w)-P*P)/(2*P*(N*N-P)). Diagonal labels/scores are zero,
as in the main-table evaluator. All ordered pairs, ties and zeros are retained.

The CUDA lookup indexes every nonnegative IEEE float32 bit pattern: no bins,
rounding, sampling, or score quantization. Searchsorted is a reference backend.
"""
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def canonical_positive_codes(edge_index, n):
    edge = edge_index.detach().cpu().long()
    edge = edge[:, edge[0] != edge[1]]
    codes = torch.unique(edge[0] * n + edge[1], sorted=True)
    reverse = torch.unique(edge[1] * n + edge[0], sorted=True)
    assert torch.equal(codes, reverse), 'Expected undirected graph'
    return codes


def row_scores(h, start, block):
    stop = min(start + block, h.shape[0])
    # Both passes use exactly the same matrix shapes and multiplication.
    scores = torch.mm(h[start:stop], h.t()).relu_()
    idx = torch.arange(stop - start, device=h.device)
    scores[idx, idx + start] = 0
    return scores


def make_exact_rank_lut(positive):
    assert positive.dtype == torch.float32 and positive.is_cuda
    p = positive.numel()
    assert 2 * p < 2**31
    bits = positive.contiguous().view(torch.int32).clamp_min(0)
    keys, counts = torch.unique(bits, sorted=True, return_counts=True)
    limit = int(keys[-1]) + 1  # sentinel for every score above largest positive
    hist = torch.zeros(limit + 1, dtype=torch.int32, device=positive.device)
    hist[keys.long()] = counts.to(torch.int32)
    rank = torch.cumsum(hist, dim=0, dtype=torch.int32)
    rank.mul_(-2).add_(2 * p).add_(hist)
    del hist, bits, keys, counts
    return rank, limit


@torch.no_grad()
def exact_auc(h, edge_index, block=256, backend='lut', output=None):
    started = time.time()
    assert h.dtype == torch.float32 and torch.isfinite(h).all()
    n = h.shape[0]
    h = F.normalize(h - h.mean(dim=1, keepdim=True), p=2, dim=1)
    codes = canonical_positive_codes(edge_index, n)
    p = codes.numel()
    assert 0 < p < n * n and 2 * p * n * n < 2**63
    positive = torch.empty(p, dtype=torch.float32, device=h.device)
    needs_sigmoid = False
    bounds = torch.searchsorted(codes, torch.arange(0, n + block, block) * n).tolist()
    for b, start in enumerate(range(0, n, block)):
        scores = row_scores(h, start, block)
        needs_sigmoid = needs_sigmoid or bool((scores > 1).any())
        lo, hi = bounds[b], bounds[b + 1]
        index = (codes[lo:hi] - start * n).to(h.device)
        positive[lo:hi] = scores.reshape(-1)[index]
        del scores, index
    # BinaryAUROC interprets values outside [0,1] as logits. Rare float32
    # overshoots near 1 must trigger the same transform, including diagonal 0.
    if needs_sigmoid:
        positive.sigmoid_()
    if output:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(positive.cpu(), output/'positive_scores.pt')
    if h.is_cuda:
        torch.cuda.synchronize()
    print(f'Exact AUC positive pass: P={p}, N*N={n*n}, seconds={time.time()-started:.2f}', flush=True)
    if backend == 'lut':
        lookup, limit = make_exact_rank_lut(positive)
    elif backend == 'search':
        ordered = positive.sort().values
    else:
        raise ValueError(backend)
    total = 0
    block_records = []
    for start in range(0, n, block):
        scores = row_scores(h, start, block)
        if needs_sigmoid:
            scores.sigmoid_()
        if backend == 'lut':
            # Nonnegative float32 bit order equals numeric order. -0 maps to 0.
            indices = scores.reshape(-1).view(torch.int32).clamp(0, limit)
            weight = lookup[indices.long()].sum(dtype=torch.int64)
            del indices
        else:
            flat = scores.reshape(-1)
            left = torch.searchsorted(ordered, flat, right=False)
            right = torch.searchsorted(ordered, flat, right=True)
            weight = (2*p-left-right).sum(dtype=torch.int64)
            del left, right
        value = int(weight)
        total += value
        record = {'start':start, 'stop':min(start+block,n), 'sum_twice_wins':value,
                  'seconds':time.time()-started}
        block_records.append(record)
        if output:
            with (output/'blocks.jsonl').open('a') as f:
                f.write(json.dumps(record)+'\n')
        if start % (block*32) == 0 or start+block >= n:
            print(f'Exact AUC rows {min(start+block,n)}/{n}, seconds={time.time()-started:.2f}', flush=True)
        del scores, weight
    numerator = total-p*p
    denominator = 2*p*(n*n-p)
    auc = numerator/denominator
    assert 0 <= auc <= 1
    result = {'raw_auc':auc, 'auc_sep':max(auc,1-auc), 'num_nodes':n,
              'positive_ordered_pairs':p, 'negative_ordered_pairs_including_diagonal':n*n-p,
              'total_pairs':n*n, 'diagonal_negatives':n,
              'sum_all_twice_wins':total, 'positive_self_contribution':p*p,
              'auc_integer_numerator':numerator, 'auc_integer_denominator':denominator,
              'seconds':time.time()-started, 'backend':backend, 'block_rows':block,
              'score_dtype':'float32', 'sampled':False, 'quantized':False,
              'binary_auroc_auto_sigmoid':needs_sigmoid}
    if output:
        (output/'exact_result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
