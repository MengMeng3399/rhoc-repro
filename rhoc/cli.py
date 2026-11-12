import argparse
import hashlib
import json
import math
import struct
import time
from pathlib import Path

from .data import PARTITIONS, build_queries, load_catalog, nested_queries, split_invocations, write_demo

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
DATASETS = ("pw", "hag")
CONTEXTS = ("1", "2", "3", "mixed")
METHODS = ("random", "popular", "ap", "greedy", "bm25")
VARIANTS = ("full", "masked-text", "no-text", "no-alignment", "no-transformer", "positional")


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def nonnegative_float(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("expected a finite nonnegative number")
    return number


def data_args(parser, default_dataset=None):
    parser.add_argument("--dataset", choices=DATASETS, required=default_dataset is None, default=default_dataset)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split-seed", type=int)


def experiment_args(parser, default_epochs=None):
    parser.add_argument("--context", choices=CONTEXTS, default="mixed")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--output", type=Path)
    if default_epochs is not None:
        parser.add_argument("--epochs", type=positive_int, default=default_epochs)
        parser.add_argument("--variant", choices=VARIANTS, default="full")
        parser.add_argument("--drop-view", choices=("id", "category", "tag", "provider"))
        parser.add_argument("--alignment-weight", type=nonnegative_float)
        parser.add_argument("--layers", type=positive_int)
        parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        parser.add_argument("--diagnostics", action="store_true")
        parser.add_argument("--bge-checkpoint")


def config_for(args):
    config = json.loads((CONFIGS / f"{args.dataset}.json").read_text(encoding="utf-8"))
    if args.split_seed is None:
        args.split_seed = config["split_seed"]
    config["split_seed"] = args.split_seed
    if getattr(args, "seeds", None) is None:
        args.seeds = [config["run_seeds"][0]] if args.command == "demo" else config["run_seeds"]
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("Each run seed must be unique.")
    if any(not 0 <= seed < 2**32 for seed in args.seeds):
        raise ValueError("Run seeds must be in [0, 2**32).")
    config["run_seeds"] = args.seeds
    return config


def queries_for(splits, config, partition, sizes, seed):
    quotas = {int(size): count for size, count in config["query_counts"][partition].items()}
    return build_queries(splits, partition, sizes, quotas, seed + PARTITIONS.index(partition))


def test_fingerprint(catalog, queries):
    digest = hashlib.sha256(json.dumps([catalog.app_ids, catalog.api_ids]).encode("utf-8"))
    for query in queries:
        values = (query.app, query.target, len(query.context), *query.context)
        digest.update(struct.pack("<" + "Q" * len(values), *values))
    return digest.hexdigest()


def require_queries(partition, queries):
    if not queries:
        raise ValueError(f"No {partition} queries are eligible for the selected context size.")


def catalog_counts(catalog):
    return {
        "applications": len(catalog.app_ids),
        "apis": len(catalog.api_ids),
        "invocations": sum(map(len, catalog.observed)),
    }


def graph_counts(views):
    return {
        view.name: {"nodes": view.adjacency.shape[0], "edges": view.adjacency._nnz() // 2}
        for view in views
    }


def counts(args):
    from collections import Counter

    from .experiment import write_report
    from .model import build_views

    config = config_for(args)
    catalog = load_catalog(args.data)
    splits = split_invocations(catalog, args.split_seed)
    query_counts = {}
    for partition in PARTITIONS:
        queries = queries_for(splits, config, partition, (1, 2, 3), args.split_seed)
        lengths = Counter(len(query.context) for query in queries)
        query_counts[partition] = {str(size): lengths[size] for size in (1, 2, 3)}
        query_counts[partition]["mixed"] = len(queries)
        del queries
    report = {
        "dataset": args.dataset,
        "split_seed": args.split_seed,
        "catalog": catalog_counts(catalog),
        "invocation_split": {key: sum(map(len, value)) for key, value in splits.items()},
        "queries": query_counts,
        "graphs": graph_counts(build_views(catalog, splits["train"], config["views"])),
        "paper_reference": {
            "catalog": config["expected_catalog"],
            "queries": config["query_counts"],
            "graphs": config["expected_graphs"],
        },
    }
    write_report(report, None)


def train(args):
    import torch

    from .experiment import Cohort, aggregate, select_device, train_one, write_report
    from .metrics import frequency_groups
    from .model import build_views
    from .text import BGEEncoder, DemoEncoder, mask_requirements

    config = config_for(args)
    demo = args.command == "demo"
    if demo:
        write_demo(args.data, args.dataset)
        config["validate_every"] = 1
        args.device = "cpu"
        torch.set_num_threads(min(torch.get_num_threads(), 2))
    device = select_device(args.device)
    if args.drop_view:
        if args.drop_view not in config["views"]:
            raise ValueError(f"Graph view {args.drop_view!r} is unavailable for {args.dataset}.")
        config["views"].remove(args.drop_view)
    if args.layers is not None:
        config["transformer_layers"] = args.layers
    if args.alignment_weight is not None:
        config["alignment_weight"] = args.alignment_weight
    if args.variant in ("no-alignment", "no-text"):
        config["alignment_weight"] = 0.0
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
    if demo and args.bge_checkpoint:
        raise ValueError("The synthetic demo uses its built-in text encoder. Use train for BGE.")

    started = time.perf_counter()
    catalog = load_catalog(args.data)
    splits = split_invocations(catalog, args.split_seed)
    sizes = (1, 2, 3) if args.context == "mixed" else (int(args.context),)
    cohorts = {}
    for partition in PARTITIONS:
        queries = queries_for(splits, config, partition, sizes, args.split_seed)
        require_queries(partition, queries)
        cohorts[partition] = Cohort.from_queries(queries)
    views = build_views(catalog, splits["train"], config["views"])
    preprocessing_seconds = time.perf_counter() - started

    text = None
    text_checkpoint = None
    if args.variant != "no-text":
        texts = catalog.requirements
        if args.variant == "masked-text":
            texts = mask_requirements(texts, catalog.api_rows, args.dataset)
        if demo:
            encoder = DemoEncoder(config["text_dim"])
            text_checkpoint = "synthetic-hashed-text"
        else:
            text_checkpoint = args.bge_checkpoint or config["checkpoint"]
            encoder = BGEEncoder(text_checkpoint, device, config["text_dim"])
        text = encoder.encode(texts, config["batch_size"])
        del encoder

    nested = None
    groups = None
    if args.diagnostics:
        nested = {
            size: Cohort.from_queries(queries)
            for size, queries in nested_queries(splits, args.split_seed + 100).items()
        }
        groups = frequency_groups(splits["train"], len(catalog.api_ids))
    runs = [
        train_one(catalog, views, cohorts, text, config, args, device, seed, nested, groups)
        for seed in args.seeds
    ]
    write_report({
        "method": "rhoc",
        "variant": args.variant,
        "synthetic": demo,
        "dataset": args.dataset,
        "context": args.context,
        "device": str(device),
        "text_checkpoint": text_checkpoint,
        "epochs_cap": args.epochs,
        "config": config,
        "catalog": catalog_counts(catalog),
        "graphs": graph_counts(views),
        "test_fingerprint": test_fingerprint(catalog, cohorts["test"].queries),
        "query_counts": {key: len(value.queries) for key, value in cohorts.items()},
        "preprocessing_seconds": preprocessing_seconds,
        "runs": runs,
        "summary": aggregate(runs),
    }, args.output)


def baseline(args):
    import torch

    from .baselines import create_scorer
    from .experiment import Cohort, aggregate, batches, metric_means, write_report
    from .metrics import frequency_groups, ranking_batch

    config = config_for(args)
    catalog = load_catalog(args.data)
    splits = split_invocations(catalog, args.split_seed)
    sizes = (1, 2, 3) if args.context == "mixed" else (int(args.context),)
    queries = queries_for(splits, config, "test", sizes, args.split_seed)
    require_queries("test", queries)
    cohort = Cohort.from_queries(queries)
    groups = frequency_groups(splits["train"], len(catalog.api_ids))
    labels = [groups[target] for target in cohort.targets.tolist()]
    group_indices = {
        group: [i for i, label in enumerate(labels) if label == group]
        for group in ("head", "middle", "tail")
    }
    runs = []
    for seed in args.seeds:
        scorer = create_scorer(args.method, catalog, splits["train"], seed)
        totals = {}
        for indices in batches(cohort, config["batch_size"]):
            scores = torch.tensor([scorer(queries[index]) for index in indices.tolist()], dtype=torch.float64)
            values = ranking_batch(scores, cohort.contexts[indices], cohort.targets[indices])
            for name, part in values.items():
                totals.setdefault(name, []).append(part)
        values = {name: torch.cat(parts) for name, parts in totals.items()}
        runs.append({
            "seed": seed,
            "metrics": metric_means(values),
            "frequency_groups": {
                group: {"queries": len(indices), "Recall@20": values["Recall@20"][indices].mean().item()}
                for group, indices in group_indices.items()
                if indices
            },
        })
    write_report({
        "method": args.method,
        "dataset": args.dataset,
        "context": args.context,
        "split_seed": args.split_seed,
        "test_fingerprint": test_fingerprint(catalog, queries),
        "runs": runs,
        "summary": aggregate(runs),
    }, args.output)


def compare(args):
    from scipy.stats import ttest_rel

    from .experiment import write_report

    reference = json.loads(args.rhoc.read_text(encoding="utf-8"))
    baselines = [json.loads(path.read_text(encoding="utf-8")) for path in args.baseline]
    seeds = {run["seed"] for run in reference["runs"]}
    if len(seeds) < 2:
        raise ValueError("A paired comparison needs at least two matching seeds.")
    for report in baselines:
        if any(report[key] != reference[key] for key in ("dataset", "context", "test_fingerprint")):
            raise ValueError("Run reports must share the same test queries and API catalog.")
        if {run["seed"] for run in report["runs"]} != seeds:
            raise ValueError("Run reports must share the same seeds.")

    results = {}
    for metric, summary in reference["summary"].items():
        strongest = max(baselines, key=lambda report: report["summary"][metric]["mean"])
        baseline_mean = strongest["summary"][metric]["mean"]
        first = {run["seed"]: run["metrics"][metric] for run in reference["runs"]}
        second = {run["seed"]: run["metrics"][metric] for run in strongest["runs"]}
        differences = {first[seed] - second[seed] for seed in seeds}
        if len(differences) == 1:
            p_value = None if differences == {0.0} else 0.0
        else:
            p_value = float(ttest_rel(
                [first[seed] for seed in sorted(seeds)], [second[seed] for seed in sorted(seeds)]
            ).pvalue)
        results[metric] = {
            "rhoc_mean": summary["mean"],
            "strongest_supplied_baseline": strongest["method"],
            "baseline_mean": baseline_mean,
            "relative_gain_percent": 100 * (summary["mean"] / baseline_mean - 1) if baseline_mean else None,
            "paired_two_sided_p": p_value,
        }
    write_report(results, None)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate RHOC for incremental cloud API selection.")
    commands = parser.add_subparsers(dest="command", required=True)
    counts_parser = commands.add_parser("counts", help="Report catalog, split, query and graph counts.")
    data_args(counts_parser)
    counts_parser.set_defaults(handler=counts)
    train_parser = commands.add_parser("train", help="Train RHOC on local preprocessed records.")
    data_args(train_parser)
    experiment_args(train_parser, default_epochs=200)
    train_parser.set_defaults(handler=train)
    demo_parser = commands.add_parser("demo", help="Run an offline smoke test with synthetic records.")
    data_args(demo_parser, default_dataset="pw")
    experiment_args(demo_parser, default_epochs=2)
    demo_parser.set_defaults(handler=train)
    baseline_parser = commands.add_parser("baseline", help="Evaluate a bundled baseline.")
    data_args(baseline_parser)
    experiment_args(baseline_parser)
    baseline_parser.add_argument("--method", choices=METHODS, required=True)
    baseline_parser.set_defaults(handler=baseline)
    compare_parser = commands.add_parser("compare", help="Compare paired runs on the same test queries.")
    compare_parser.add_argument("--rhoc", required=True, type=Path)
    compare_parser.add_argument("--baseline", action="append", required=True, type=Path)
    compare_parser.set_defaults(handler=compare)
    args = parser.parse_args()
    try:
        args.handler(args)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")
