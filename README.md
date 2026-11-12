# RHOC

Reference implementation for requirement-aware incremental cloud API selection.

## Data and configuration

See the [PW](configs/pw.json) and [HAG](configs/hag.json) configurations for hyperparameters, query quotas, default seeds, and frozen BGE checkpoints. Supply preprocessed records locally via `--data`.

- `applications.jsonl` with `id` and `requirement`
- `apis.jsonl` with `id`, `name`, `description`, and `category` (plus `tag` and `provider` for PW)
- `invocations.csv` with `application_id` and `api_id`

Matching the reported scores requires the original data, splits, and baseline configurations.

## Query construction and evaluation

[`rhoc/data.py`](rhoc/data.py) splits invocation records 8:1:1 before generating queries. Contexts and co-invocation graphs use training records. Targets come from their respective partitions. The mixed cohort combines context sizes 1, 2, and 3. The `counts` command reports catalog, graph, and query counts.

Rank the full API catalog after excluding context APIs. The two negatives per positive are for ranking loss. Training negatives are resampled each epoch. Validation and test negatives are fixed. Report mean and standard deviation over five seeds for Recall@10/20 and NDCG@10/20.

## Run

Use Python 3.10 or later.

```bash
python -m pip install -r requirements.txt
python -m rhoc counts --dataset pw --data data/pw
python -m rhoc train --dataset pw --data data/pw --context mixed --output runs/pw
```

Use `--dataset hag --data data/hag` for HAG. Validation runs every five epochs, with early stopping after ten consecutive checks without improvement.

- **Ablations.** Use `--variant` with `masked-text`, `no-text`, `no-alignment`, `no-transformer`, or `positional`. Use `--drop-view` to remove a graph view.
- **Diagnostics.** `--diagnostics` reports performance by API frequency, fixed-target contexts of sizes 1–6, and 20 permutations of six-API contexts.
- **Baselines.** `baseline` runs Random, Popular, AP, Greedy, or BM25. `compare` tests matching seeds with a paired two-sided t-test.
