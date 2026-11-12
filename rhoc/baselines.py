import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from functools import lru_cache
from itertools import combinations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .data import Catalog, Query


BASELINES = ("random", "popular", "ap", "greedy", "bm25")
_HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_WORD = re.compile(r"[^\W_]+")


def _tokenize(text: str) -> list[str]:
    return _WORD.findall(_HAN.sub(r" \g<0> ", text.casefold()))


def _bm25(
    catalog: "Catalog",
    *,
    k1: float = 1.2,
    b: float = 0.75,
) -> Callable[["Query"], list[float]]:
    documents = [Counter(_tokenize(row["description"])) for row in catalog.api_rows]
    lengths = [sum(document.values()) for document in documents]
    n_apis = len(documents)
    average_length = sum(lengths) / n_apis or 1.0
    normalization = [k1 * (1 - b + b * length / average_length) for length in lengths]
    index = defaultdict(list)
    for api, document in enumerate(documents):
        for term, frequency in document.items():
            index[term].append((api, frequency))

    idf = {
        term: math.log1p((n_apis - len(postings) + 0.5) / (len(postings) + 0.5))
        for term, postings in index.items()
    }

    @lru_cache(maxsize=128)
    def scores_for_app(app: int) -> list[float]:
        scores = [0.0] * n_apis
        for term in dict.fromkeys(_tokenize(catalog.requirements[app])):
            if term not in index:
                continue
            for api, frequency in index[term]:
                scores[api] += idf[term] * (k1 + 1) * frequency / (
                    frequency + normalization[api]
                )
        return scores

    return lambda query: scores_for_app(query.app)


def _training_postings(train: Sequence[Sequence[int]], n_apis: int) -> list[set[int]]:
    postings = [set() for _ in range(n_apis)]
    for app, row in enumerate(train):
        for api in row:
            postings[api].add(app)
    return postings


def _association_patterns(
    train: Sequence[Sequence[int]],
    postings: list[set[int]],
    *,
    min_joint_support: int = 2,
) -> Callable[["Query"], list[float]]:
    @lru_cache(maxsize=1024)
    def scores_for_context(context: tuple[int, ...]) -> list[float]:
        scores = [0.0] * len(postings)
        for size in range(1, len(context) + 1):
            for antecedent in combinations(context, size):
                matches = postings[antecedent[0]].intersection(
                    *(postings[api] for api in antecedent[1:])
                )
                if len(matches) < min_joint_support:
                    continue
                support = Counter(api for app in matches for api in train[app])
                for api, count in support.items():
                    if count >= min_joint_support:
                        scores[api] = max(scores[api], count / len(matches))
        return scores

    return lambda query: scores_for_context(tuple(sorted(query.context)))


def _greedy_coverage(postings: list[set[int]]) -> Callable[["Query"], list[float]]:
    @lru_cache(maxsize=1024)
    def scores_for_context(context: tuple[int, ...]) -> list[float]:
        covered = set().union(*(postings[api] for api in context))
        return [float(len(apps - covered)) for apps in postings]

    return lambda query: scores_for_context(tuple(sorted(query.context)))


def create_scorer(
    name: str,
    catalog: "Catalog",
    train: Sequence[Sequence[int]],
    seed: int,
) -> Callable[["Query"], list[float]]:
    """Build a catalog scorer using the training split and catalog metadata.

    AP uses maximum rule confidence over nonempty context subsets and
    requires joint support in two training applications. Greedy measures
    marginal coverage of training applications. BM25 uses descriptions,
    case-folded word tokens and Han unigrams, with k1=1.2 and b=0.75.
    """
    name = name.lower()
    n_apis = len(catalog.api_ids)
    if name == "random":
        rng = random.Random(seed)

        def random_scores(query: "Query") -> list[float]:
            scores = [float(index) for index in range(n_apis)]
            rng.shuffle(scores)
            return scores

        return random_scores

    if name == "bm25":
        return _bm25(catalog)

    if name == "popular":
        postings = _training_postings(train, n_apis)
        scores = [float(len(apps)) for apps in postings]
        return lambda query: scores
    if name == "ap":
        return _association_patterns(train, _training_postings(train, n_apis))
    if name == "greedy":
        return _greedy_coverage(_training_postings(train, n_apis))
    raise ValueError(f"Unknown baseline {name!r}. Choose from {', '.join(BASELINES)}.")
