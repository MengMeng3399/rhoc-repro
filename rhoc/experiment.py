import itertools
import json
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .data import sample_negatives
from .metrics import frequency_groups, ranking_batch
from .model import RHOC


@dataclass
class Cohort:
    queries: list
    apps: torch.Tensor
    contexts: torch.Tensor
    targets: torch.Tensor

    @classmethod
    def from_queries(cls, queries):
        width = max((len(query.context) for query in queries), default=0)
        return cls(
            queries,
            torch.tensor([query.app for query in queries], dtype=torch.long),
            torch.tensor(
                [query.context + (-1,) * (width - len(query.context)) for query in queries],
                dtype=torch.long,
            ).reshape(len(queries), width),
            torch.tensor([query.target for query in queries], dtype=torch.long),
        )


def select_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Select --device cpu or install a CUDA-enabled PyTorch build.")
    return torch.device(name)


def batches(cohort, batch_size, shuffle_seed=None):
    size = len(cohort.queries)
    indices = torch.arange(size)
    if shuffle_seed is not None:
        generator = torch.Generator().manual_seed(shuffle_seed)
        indices = torch.randperm(size, generator=generator)
    return indices.split(batch_size)


def negative_tensors(catalog, cohort, seed, config, align):
    ranking = sample_negatives(catalog, cohort.queries, seed, config["ranking_negatives"])
    alignment = (
        sample_negatives(catalog, cohort.queries, seed + 1, config["alignment_negatives"])
        if align
        else None
    )
    return (
        torch.tensor(ranking, dtype=torch.long),
        torch.tensor(alignment, dtype=torch.long).flatten() if align else None,
    )


def epoch_loss(model, cohort, text, negatives, config, device, optimizer=None, shuffle_seed=None):
    training = optimizer is not None
    model.train(training)
    batch_size = config["batch_size"]
    ranking, alignment = negatives
    total = torch.zeros((), device=device)
    with torch.set_grad_enabled(training):
        api_embeddings = None if training else model.api_embeddings()
        for indices in batches(cohort, batch_size, shuffle_seed):
            if training:
                optimizer.zero_grad(set_to_none=True)
                api_embeddings = model.api_embeddings()
            loss = model.loss(
                api_embeddings,
                cohort.contexts[indices].to(device),
                None if text is None else text[cohort.apps[indices]],
                cohort.targets[indices].to(device),
                ranking[indices].to(device),
                None if alignment is None else alignment[indices].to(device),
                temperature=config["temperature"],
                alignment_weight=config["alignment_weight"],
            )
            if training:
                loss.backward()
                optimizer.step()
            total += loss.detach() * len(indices)
    return (total / len(cohort.queries)).item()


def metric_means(values):
    return {name: float(scores.mean()) for name, scores in values.items()}


@torch.inference_mode()
def evaluate(model, api_embeddings, cohort, text, batch_size, device, permutation=None):
    model.eval()
    results = {}
    for indices in batches(cohort, batch_size):
        contexts = cohort.contexts[indices].to(device)
        if permutation is not None:
            contexts = contexts[:, permutation]
        vectors = model.query_embeddings(
            api_embeddings, contexts, None if text is None else text[cohort.apps[indices]]
        )
        values = ranking_batch(vectors @ api_embeddings.T, contexts, cohort.targets[indices].to(device))
        for name, scores in values.items():
            results.setdefault(name, []).append(scores.cpu())
    return {name: torch.cat(parts) for name, parts in results.items()}


@torch.inference_mode()
def permutation_diagnostics(model, api_embeddings, cohort, text, batch_size, device, seed):
    order = tuple(range(6))
    permutations = list(itertools.permutations(order))[1:]
    chosen = random.Random(seed).sample(permutations, 20)
    max_score_change = torch.zeros((), device=device)
    deltas = {"Recall@10": [], "NDCG@10": []}
    size = len(cohort.queries)
    for permutation in chosen:
        changes = {name: torch.zeros((), device=device) for name in deltas}
        for indices in batches(cohort, batch_size):
            contexts = cohort.contexts[indices].to(device)
            target = cohort.targets[indices].to(device)
            text_vectors = None if text is None else text[cohort.apps[indices]]
            original = model.query_embeddings(api_embeddings, contexts, text_vectors) @ api_embeddings.T
            permuted = model.query_embeddings(api_embeddings, contexts[:, permutation], text_vectors)
            permuted = permuted @ api_embeddings.T
            difference = (original - permuted).abs()
            difference.scatter_(1, contexts, 0)
            max_score_change = torch.maximum(max_score_change, difference.max())
            initial = ranking_batch(original, contexts, target, ks=(10,))
            reordered = ranking_batch(permuted, contexts, target, ks=(10,))
            for name in changes:
                changes[name] += (reordered[name] - initial[name]).sum()
        for name, change in changes.items():
            deltas[name].append(abs(change.item()) * 100 / size)
    return {
        "queries": size,
        "permutations": len(chosen),
        "max_candidate_score_change": max_score_change.item(),
        "mean_absolute_metric_change_percentage_points": {
            name: statistics.mean(changes) for name, changes in deltas.items()
        },
    }


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_model(views, config, variant):
    return RHOC(
        views,
        dim=config["dim"],
        hidden=config["text_hidden"],
        text_dim=config["text_dim"],
        heads=config["attention_heads"],
        layers=config["transformer_layers"],
        dropout=config["dropout"],
        text=variant != "no-text",
        transformer=variant != "no-transformer",
        positional=variant == "positional",
    )


def train_one(catalog, views, cohorts, text, config, args, device, seed, nested=None, groups=None):
    torch.manual_seed(seed)
    random.seed(seed)
    align = text is not None and config["alignment_weight"] != 0
    model = build_model(views, config, args.variant).to(device)
    if text is not None:
        text = text.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    validation_negatives = negative_tensors(catalog, cohorts["validation"], args.split_seed + 200, config, align)
    test_negatives = negative_tensors(catalog, cohorts["test"], args.split_seed + 300, config, align)
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    train_times = []
    peak_bytes = 0
    measure_gpu = device.type == "cuda"
    for epoch in range(1, args.epochs + 1):
        synchronize(device)
        if measure_gpu:
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        train_negatives = negative_tensors(catalog, cohorts["train"], seed * 100_000 + epoch * 10, config, align)
        loss = epoch_loss(
            model, cohorts["train"], text, train_negatives, config, device, optimizer, seed + epoch
        )
        synchronize(device)
        train_times.append(time.perf_counter() - start)
        if measure_gpu:
            peak_bytes = max(peak_bytes, torch.cuda.max_memory_allocated(device))
        if epoch % config["validate_every"] != 0 and epoch != args.epochs:
            continue
        val_loss = epoch_loss(model, cohorts["validation"], text, validation_negatives, config, device)
        print(f"seed {seed} epoch {epoch} train loss {loss:.6f} validation loss {val_loss:.6f}", flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    if best_state is None:
        raise ValueError("Training did not produce a finite validation loss.")

    model.load_state_dict(best_state)
    model.eval()
    test_loss = epoch_loss(model, cohorts["test"], text, test_negatives, config, device)
    with torch.inference_mode():
        api_embeddings = model.api_embeddings()
    scores = evaluate(model, api_embeddings, cohorts["test"], text, config["batch_size"], device)
    result = {
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_loss": best_loss,
        "test_loss": test_loss,
        "metrics": metric_means(scores),
        "mean_train_seconds_per_epoch": statistics.mean(train_times),
        "peak_training_gpu_gib": peak_bytes / 2**30 if measure_gpu else None,
    }
    if groups:
        labels = [groups[target] for target in cohorts["test"].targets.tolist()]
        result["frequency_groups"] = {
            group: {
                "queries": len(selected),
                "Recall@20": scores["Recall@20"][selected].mean().item(),
            }
            for group in ("head", "middle", "tail")
            if (selected := [index for index, label in enumerate(labels) if label == group])
        }
    if nested and len(nested[6].queries):
        result["nested_contexts"] = {
            size: {
                "queries": len(cohort.queries),
                **metric_means(evaluate(model, api_embeddings, cohort, text, config["batch_size"], device)),
            }
            for size, cohort in nested.items()
        }
        result["permutation"] = permutation_diagnostics(
            model, api_embeddings, nested[6], text, config["batch_size"], device, args.split_seed
        )
    if args.output:
        torch.save(best_state, args.output / f"seed-{seed}.pt")
    return result


def aggregate(runs):
    return {
        name: {
            "mean": statistics.mean(run["metrics"][name] for run in runs),
            "std": statistics.stdev(run["metrics"][name] for run in runs) if len(runs) > 1 else None,
        }
        for name in runs[0]["metrics"]
    }


def write_report(report, output):
    serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if output:
        output.mkdir(parents=True, exist_ok=True)
        (output / "summary.json").write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
