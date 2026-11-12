"""RHOC graph views, requirement-conditioned query encoder, and learning objective."""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GraphView:
    name: str
    adjacency: Tensor
    api_nodes: Tensor


def _metadata_values(raw: str | list[str] | None) -> set[str]:
    values = [raw] if isinstance(raw, str) else (raw or [])
    return {value.strip() for value in values if value and value.strip()}


def build_views(
    catalog, train: Iterable[tuple[int, ...]], names: Iterable[str]
) -> list[GraphView]:
    """Build one weighted graph per view from training invocations only."""
    view_names = [name.casefold() for name in names]
    if not view_names or len(set(view_names)) != len(view_names):
        raise ValueError("At least one graph view is required, with each view named once")

    training_apis = [sorted(set(invoked)) for invoked in train]
    views = []
    for name in view_names:
        if name == "id":
            nodes = [(api_index,) for api_index in range(len(catalog.api_ids))]
            node_count = len(nodes)
        elif name in {"category", "tag", "provider"}:
            values = [_metadata_values(row.get(name)) for row in catalog.api_rows]
            distinct_values = sorted({value for row in values for value in row})
            value_ids = {value: index for index, value in enumerate(distinct_values)}
            nodes = [tuple(sorted(value_ids[value] for value in row)) for row in values]
            node_count = len(value_ids)
        else:
            raise ValueError(f"Unknown graph view {name!r}")
        if node_count == 0:
            raise ValueError(f"Graph view {name!r} has no values in the API catalog")

        edge_counts: Counter[tuple[int, int]] = Counter()
        for invoked in training_apis:
            # Count each unordered attribute pair at most once per application.
            app_edges = {
                (min(u, v), max(u, v))
                for a, b in combinations(invoked, 2)
                for u in nodes[a]
                for v in nodes[b]
                if u != v
            }
            edge_counts.update(app_edges)

        rows = []
        cols = []
        weights = []
        for (u, v), count in edge_counts.items():
            rows.extend((u, v))
            cols.extend((v, u))
            weights.extend((count, count))
        indices = torch.tensor([rows, cols], dtype=torch.long)
        edge_weights = torch.tensor(weights, dtype=torch.float32)
        degree = torch.bincount(indices[0], weights=edge_weights, minlength=node_count)
        normalized_weights = edge_weights / degree[indices[0]]
        adjacency = torch.sparse_coo_tensor(
            indices, normalized_weights, (node_count, node_count), check_invariants=True
        ).coalesce()

        max_nodes = max(map(len, nodes))
        api_nodes = torch.tensor(
            [list(row) + [-1] * (max_nodes - len(row)) for row in nodes],
            dtype=torch.long,
        )
        views.append(GraphView(name, adjacency, api_nodes))
    return views


def _masked_mean(vectors: Tensor, valid: Tensor) -> Tensor:
    total = vectors.masked_fill(~valid.unsqueeze(-1), 0).sum(dim=1)
    return total / valid.sum(dim=1, keepdim=True).clamp_min(1)


class _ViewEncoder(nn.Module):
    def __init__(self, view: GraphView, dim: int):
        super().__init__()
        num_nodes = view.adjacency.shape[0]
        adjacency = view.adjacency
        # Dense kernels are more efficient for the small, high-density views.
        if adjacency._nnz() * 8 >= num_nodes * num_nodes:
            adjacency = adjacency.to_dense()
        self.register_buffer("adjacency", adjacency, persistent=False)
        self.register_buffer("api_nodes", view.api_nodes.clamp_min(0), persistent=False)
        self.register_buffer("node_mask", view.api_nodes.ge(0), persistent=False)
        self.node_vectors = nn.Embedding(num_nodes, dim)
        self.first = nn.Linear(2 * dim, dim)
        self.second = nn.Linear(2 * dim, dim)

    def forward(self) -> Tensor:
        nodes = self.node_vectors.weight
        neighbors = self.adjacency @ nodes
        nodes = F.relu(self.first(torch.cat((nodes, neighbors), dim=-1)))
        neighbors = self.adjacency @ nodes
        nodes = self.second(torch.cat((nodes, neighbors), dim=-1))
        return _masked_mean(nodes[self.api_nodes], self.node_mask)


class RHOC(nn.Module):
    def __init__(
        self,
        views: list[GraphView],
        *,
        dim: int,
        hidden: int = 512,
        text_dim: int = 768,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
        text: bool = True,
        transformer: bool = True,
        positional: bool = False,
    ):
        super().__init__()
        if not views:
            raise ValueError("RHOC requires at least one graph view")
        if transformer and layers < 1:
            raise ValueError("At least one layer is required for the Transformer")

        self.view_encoders = nn.ModuleList(_ViewEncoder(view, dim) for view in views)
        self.register_buffer(
            "available_views",
            torch.stack([view.api_nodes.ge(0).any(dim=1) for view in views], dim=1),
            persistent=False,
        )
        self.view_attention = nn.Sequential(
            nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1, bias=False)
        )
        self.text_projection = (
            nn.Sequential(
                nn.Linear(text_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, dim),
            )
            if text
            else None
        )
        self.encoder_layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=256,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            for _ in range(layers if transformer else 0)
        )
        self.positional = positional

    def api_embeddings(self) -> Tensor:
        embeddings = torch.stack([view() for view in self.view_encoders], dim=1)
        logits = self.view_attention(embeddings).squeeze(-1)
        # Missing attributes contribute no vector and receive no weight when a view is present.
        logits = logits.masked_fill(~self.available_views, torch.finfo(logits.dtype).min)
        weights = logits.softmax(dim=1)
        return (embeddings * weights.unsqueeze(-1)).sum(dim=1)

    def _requirement(self, text_vectors: Tensor | None) -> Tensor | None:
        if self.text_projection is None:
            return None
        if text_vectors is None:
            raise ValueError("Requirement vectors are needed when text is enabled")
        return F.normalize(self.text_projection(text_vectors), dim=-1)

    def _encode_query(
        self, api_embeddings: Tensor, context_ids: Tensor, requirement: Tensor | None
    ) -> Tensor:
        valid = context_ids.ge(0)
        tokens = api_embeddings[context_ids.clamp_min(0)]
        tokens = tokens.masked_fill(~valid.unsqueeze(-1), 0)
        if requirement is not None:
            tokens = torch.cat((tokens, requirement.unsqueeze(1)), dim=1)
            valid = F.pad(valid, (0, 1), value=True)

        if self.encoder_layers:
            if self.positional:
                # Sinusoidal positions number real tokens, skipping batch padding.
                positions = valid.long().cumsum(dim=1) - 1
                frequencies = torch.arange(0, tokens.shape[-1], 2, device=tokens.device)
                inverse_frequencies = 10000 ** (-frequencies.float() / tokens.shape[-1])
                angles = positions.unsqueeze(-1) * inverse_frequencies
                encodings = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)
                tokens = tokens + encodings[..., : tokens.shape[-1]].to(tokens.dtype)
            for layer in self.encoder_layers:
                tokens = layer(tokens, src_key_padding_mask=~valid)
            if requirement is not None:
                return tokens[:, -1]

        # Mean pooling covers the no-Transformer and no-text variants.
        return _masked_mean(tokens, valid)

    def query_embeddings(
        self, api_embeddings: Tensor, context_ids: Tensor, text_vectors: Tensor | None
    ) -> Tensor:
        return self._encode_query(api_embeddings, context_ids, self._requirement(text_vectors))

    def loss(
        self,
        api_embeddings: Tensor,
        context_ids: Tensor,
        text_vectors: Tensor | None,
        positive_ids: Tensor,
        negative_ids: Tensor,
        alignment_negative_ids: Tensor | None,
        *,
        temperature: float,
        alignment_weight: float,
    ) -> Tensor:
        requirement = self._requirement(text_vectors)
        query = self._encode_query(api_embeddings, context_ids, requirement)
        positives = api_embeddings[positive_ids]
        negatives = api_embeddings[negative_ids]
        positive_scores = (query * positives).sum(dim=-1, keepdim=True)
        negative_scores = (query.unsqueeze(1) * negatives).sum(dim=-1)
        ranking_loss = F.softplus(negative_scores - positive_scores).mean()
        if requirement is None or alignment_weight == 0:
            return ranking_loss
        if alignment_negative_ids is None or temperature <= 0:
            raise ValueError("Contrastive alignment needs negative IDs and a positive temperature")

        positive_similarity = F.cosine_similarity(requirement, positives, dim=-1)
        negative_similarity = F.cosine_similarity(
            requirement, api_embeddings[alignment_negative_ids], dim=-1
        )
        scaled_difference = (negative_similarity - positive_similarity) / temperature
        alignment_loss = F.softplus(scaled_difference).mean()
        return ranking_loss + alignment_weight * alignment_loss
