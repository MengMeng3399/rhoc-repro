from collections import Counter
from collections.abc import Iterable, Sequence

import torch
from torch import Tensor


def ranking_batch(
    scores: Tensor,
    context_ids: Tensor,
    targets: Tensor,
    ks: Iterable[int] = (10, 20),
) -> dict[str, Tensor]:
    """Return per-query metrics for single targets over the full catalog.

    Context rows use -1 for padding. Ties are ordered by catalog index.
    """
    target_scores = scores.gather(1, targets[:, None])
    catalog_indices = torch.arange(scores.size(1), device=scores.device)
    ahead = (scores > target_scores) | (
        (scores == target_scores) & (catalog_indices < targets[:, None])
    )

    query_indices = torch.arange(scores.size(0), device=scores.device)[:, None]
    query_indices = query_indices.expand_as(context_ids)
    present = context_ids >= 0
    ahead[query_indices[present], context_ids[present]] = False

    rank = ahead.sum(dim=1) + 1
    discount = torch.log2(rank.float() + 1).reciprocal()
    results = {}
    for cutoff in ks:
        if cutoff < 1:
            raise ValueError("Ranking cutoffs must be positive.")
        hit = (rank <= cutoff).float()
        results[f"Recall@{cutoff}"] = hit
        results[f"NDCG@{cutoff}"] = hit * discount
    return results


def frequency_groups(
    train: Sequence[Sequence[int]], n_apis: int
) -> tuple[str, ...]:
    counts = Counter(api for row in train for api in set(row))
    ranked = sorted(range(n_apis), key=lambda api: (-counts[api], api))
    head_end = n_apis * 5 // 100
    middle_end = n_apis * 20 // 100

    labels = ["tail"] * n_apis
    for api in ranked[:head_end]:
        labels[api] = "head"
    for api in ranked[head_end:middle_end]:
        labels[api] = "middle"
    return tuple(labels)
