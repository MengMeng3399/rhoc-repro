"""Catalog I/O and the invocation-level split used for RHOC evaluation."""

from __future__ import annotations

import csv
import io
import json
import random
from bisect import bisect_right
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import combinations
from math import comb
from pathlib import Path
from sys import maxsize


PARTITIONS = ("train", "validation", "test")
METADATA_FIELDS = ("category", "tag", "provider")


@dataclass(frozen=True, slots=True)
class Query:
    app: int
    context: tuple[int, ...]
    target: int


@dataclass(slots=True)
class Catalog:
    app_ids: list[str]
    requirements: list[str]
    api_ids: list[str]
    api_rows: list[dict]
    observed: list[tuple[int, ...]]


def _load_rows(path: Path, text_fields: tuple[str, ...]) -> list[dict]:
    rows = {}
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            location = f"{path}:{line_number}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{location} contains invalid JSON: {error.msg}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{location} must contain a JSON object")
            row_id = row.get("id")
            if not isinstance(row_id, str) or not row_id.strip():
                raise ValueError(f"{location} needs a nonempty string id")
            if row_id in rows:
                raise ValueError(f"{location} repeats id {row_id!r}")
            for field in text_fields:
                if not isinstance(row.get(field), str):
                    raise ValueError(f"{location} needs a string {field!r} field")
            if "name" in text_fields:
                for field in METADATA_FIELDS:
                    values = row.get(field, [])
                    if isinstance(values, str):
                        values = [values]
                    if not isinstance(values, list) or any(
                        not isinstance(value, str) for value in values
                    ):
                        raise ValueError(
                            f"{location} field {field!r} must be a string or list of strings"
                        )
                    row[field] = sorted({value.strip() for value in values if value.strip()})
            rows[row_id] = row
    if not rows:
        raise ValueError(f"{path} has no records")
    return [rows[row_id] for row_id in sorted(rows)]


def load_catalog(root: Path) -> Catalog:
    """Load a preprocessed catalog, assigning integer indexes in sorted ID order."""
    root = Path(root)
    apps = _load_rows(root / "applications.jsonl", ("requirement",))
    apis = _load_rows(root / "apis.jsonl", ("name", "description"))
    app_ids = [row["id"] for row in apps]
    api_ids = [row["id"] for row in apis]
    app_index = {value: index for index, value in enumerate(app_ids)}
    api_index = {value: index for index, value in enumerate(api_ids)}
    observed = [set() for _ in apps]

    path = root / "invocations.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"application_id", "api_id"}.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path} needs application_id and api_id columns")
        for row in reader:
            app_id = row["application_id"]
            api_id = row["api_id"]
            if app_id not in app_index or api_id not in api_index:
                raise ValueError(
                    f"{path}:{reader.line_num} references an unknown application/API "
                    f"pair ({app_id!r}, {api_id!r})"
                )
            observed[app_index[app_id]].add(api_index[api_id])
    if not any(observed):
        raise ValueError(f"{path} has no invocation records")
    return Catalog(
        app_ids=app_ids,
        requirements=[row["requirement"] for row in apps],
        api_ids=api_ids,
        api_rows=apis,
        observed=[tuple(sorted(values)) for values in observed],
    )


def split_invocations(catalog: Catalog, seed: int) -> dict[str, list[tuple[int, ...]]]:
    """Shuffle unique relations globally. Round each held-out size to the nearest tenth."""
    records = [(app, api) for app, apis in enumerate(catalog.observed) for api in apis]
    random.Random(seed).shuffle(records)
    holdout = (len(records) + 5) // 10
    train_end = len(records) - 2 * holdout
    validation_end = len(records) - holdout
    splits = {}
    for partition, start, end in zip(
        PARTITIONS,
        (0, train_end, validation_end),
        (train_end, validation_end, len(records)),
    ):
        by_app = [[] for _ in catalog.app_ids]
        for index in range(start, end):
            app, api = records[index]
            by_app[app].append(api)
        splits[partition] = [tuple(sorted(apis)) for apis in by_app]
    return splits


def _contexts_at_positions(
    apis: tuple[int, ...], size: int, positions: Iterable[int]
) -> Iterator[tuple[int, ...]]:
    """Unrank lexicographic combinations without walking skipped combinations."""
    for position in positions:
        context = []
        first = 0
        for remaining in range(size, 0, -1):
            for index in range(first, len(apis) - remaining + 1):
                block = comb(len(apis) - index - 1, remaining - 1)
                if position < block:
                    context.append(apis[index])
                    first = index + 1
                    break
                position -= block
        yield tuple(context)


def _sample_positions(total: int, count: int, rng: random.Random) -> list[int]:
    if total <= maxsize:
        return sorted(rng.sample(range(total), count))
    # Floyd's algorithm also handles combinatorial totals beyond len(range(...)).
    selected = set()
    for last in range(total - count, total):
        value = rng.randrange(last + 1)
        selected.add(last if value in selected else value)
    return sorted(selected)


def build_queries(
    splits: dict[str, list[tuple[int, ...]]],
    partition: str,
    context_sizes: Iterable[int],
    quotas: dict[int, int],
    seed: int,
) -> list[Query]:
    """Build single-target queries using training contexts and partition-specific targets.

    Quotas apply globally per context size. An unspecified quota retains all eligible
    queries, as do one-API training and validation settings regardless of the quota.
    """
    train = splits["train"]
    targets = splits[partition]
    queries = []
    for size in dict.fromkeys(context_sizes):
        if size < 1:
            raise ValueError("context sizes must be positive")
        quota = quotas.get(size)
        if quota is not None and quota < 0:
            raise ValueError(f"quota for context size {size} cannot be negative")
        if size == 1 and partition in {"train", "validation"}:
            quota = None

        groups = []
        total = 0
        for app, (train_apis, target_apis) in enumerate(zip(train, targets, strict=True)):
            for target in target_apis:
                context_apis = train_apis
                if partition == "train":
                    context_apis = tuple(api for api in train_apis if api != target)
                if len(context_apis) < size:
                    continue
                group_size = comb(len(context_apis), size)
                groups.append((app, context_apis, target, total, total + group_size))
                total += group_size

        if quota is None or quota >= total:
            for app, apis, target, _, _ in groups:
                queries.extend(Query(app, context, target) for context in combinations(apis, size))
            continue

        positions = _sample_positions(total, quota, random.Random(f"{seed}:{size}"))
        cursor = 0
        for app, apis, target, start, end in groups:
            if cursor == len(positions):
                break
            stop = bisect_right(positions, end - 1, lo=cursor)
            if stop > cursor:
                local = (position - start for position in positions[cursor:stop])
                contexts = _contexts_at_positions(apis, size, local)
                queries.extend(Query(app, context, target) for context in contexts)
            cursor = stop
    return queries


def nested_queries(
    splits: dict[str, list[tuple[int, ...]]], seed: int
) -> dict[int, list[Query]]:
    """Use the same eligible test targets at each context size from one through six."""
    rng = random.Random(seed)
    queries = {size: [] for size in range(1, 7)}
    for app, (train_apis, targets) in enumerate(
        zip(splits["train"], splits["test"], strict=True)
    ):
        if len(train_apis) < 6:
            continue
        for target in targets:
            order = rng.sample(train_apis, 6)
            for size, group in queries.items():
                group.append(Query(app, tuple(order[:size]), target))
    return queries


def sample_negatives(
    catalog: Catalog, queries: Iterable[Query], seed: int, count: int
) -> list[tuple[int, ...]]:
    """Sample distinct unobserved APIs uniformly for each query's loss."""
    if count < 0:
        raise ValueError("negative sample count cannot be negative")
    rng = random.Random(seed)
    catalog_size = len(catalog.api_ids)
    # Subtract each blocked API's position to map dense unobserved ranks back to IDs.
    blocked = [
        tuple(api - offset for offset, api in enumerate(observed))
        for observed in catalog.observed
    ]
    samples = []
    for query in queries:
        excluded = blocked[query.app]
        available = catalog_size - len(excluded)
        if count > available:
            raise ValueError(
                f"application {catalog.app_ids[query.app]!r} has only {available} "
                f"unobserved APIs; cannot sample {count} distinct negatives"
            )
        ranks = rng.sample(range(available), count)
        samples.append(tuple(rank + bisect_right(excluded, rank) for rank in ranks))
    return samples


def _json_lines(rows: list[dict]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def write_demo(root: Path, dataset: str) -> None:
    """Write a small deterministic synthetic catalog with PW- or HAG-like fields."""
    dataset = dataset.lower()
    if dataset not in {"pw", "hag"}:
        raise ValueError("demo dataset must be 'pw' or 'hag'")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    topics = [
        ("Travel", "旅行"),
        ("Weather", "天气"),
        ("Shipping", "物流"),
        ("Events", "活动"),
        ("Education", "教育"),
        ("Media", "媒体"),
    ]
    functions = [
        ("Search", "搜索"),
        ("Alerts", "提醒"),
        ("Lookup", "查询"),
        ("Reports", "报告"),
        ("Scheduling", "排程"),
        ("Export", "导出"),
        ("Tracking", "追踪"),
        ("Summary", "摘要"),
    ]
    language = int(dataset == "hag")
    apis = []
    apps = []
    invocations = io.StringIO(newline="")
    writer = csv.writer(invocations, lineterminator="\n")
    writer.writerow(("application_id", "api_id"))
    for topic_index, topics_by_language in enumerate(topics):
        topic = topics_by_language[language]
        for function_index, functions_by_language in enumerate(functions):
            function = functions_by_language[language]
            api_id = f"synthetic-{dataset}-api-{topic_index:02d}-{function_index:02d}"
            row = {
                "id": api_id,
                "name": f"{topic} {function}",
                "description": (
                    f"演示接口，提供{topic}{function}。"
                    if language
                    else f"Synthetic API for {function.lower()} in {topic.lower()} workflows."
                ),
                "category": topic,
            }
            if dataset == "pw":
                row["tag"] = [function, topic]
                row["provider"] = f"Synthetic provider {function_index % 3}"
            apis.append(row)

        for variant in range(3):
            app_id = f"synthetic-{dataset}-app-{topic_index:02d}-{variant:02d}"
            apps.append(
                {
                    "id": app_id,
                    "requirement": (
                        f"演示应用{variant}，用于{topic}搜索、提醒、报告和数据导出。"
                        if language
                        else f"Synthetic {topic.lower()} app {variant} for search, "
                        "alerts, reports, and data export."
                    ),
                }
            )
            # Six same-topic services plus two services shared across workflows.
            for offset in range(6):
                api_id = f"synthetic-{dataset}-api-{topic_index:02d}-{(variant + offset) % 8:02d}"
                writer.writerow((app_id, api_id))
            shared_topic = (topic_index + 1) % len(topics)
            for function_index in (1, 3):
                writer.writerow(
                    (app_id, f"synthetic-{dataset}-api-{shared_topic:02d}-{function_index:02d}")
                )

    contents = {
        root / "applications.jsonl": _json_lines(apps),
        root / "apis.jsonl": _json_lines(apis),
        root / "invocations.csv": invocations.getvalue(),
    }
    for path, content in contents.items():
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"demo generation would replace existing data at {path}")
    for path, content in contents.items():
        if not path.exists():
            with path.open("x", encoding="utf-8", newline="") as stream:
                stream.write(content)
